
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from easydict import EasyDict as edict
from ...utils import *
from ...models import BaseModel
from ...contrib import *


class CQL(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        
        rl_params = self.policy.get_rl_params()

        self.policy_optim = Adam(
            rl_params['policy'],
            # lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )


        if rl_params['consistency']:
            self.consistency_optims = {  name : {  
                'optimizer' : Adam( args['params'], lr = args['lr'] ),
                'metric' : args['metric']

            }   for name, args in rl_params['consistency'].items() }


            scheduler_params = edict(
                factor = 0.5,
                patience = 10, # 4 epsisode
                verbose= True,
            )


            self.consistency_schedulers = {
                k : Scheduler_Helper(v['optimizer'], **scheduler_params, module_name = k) for k, v in self.consistency_optims.items()
            }

            self.schedulers_metric = {
                k : v['metric'] for k, v in self.consistency_optims.items()
            }

            self.consistency_meters = { k : AverageMeter() for k in self.consistency_optims.keys()}

                
        # Alpha
        if isinstance(self.init_alpha, float):
            self.init_alpha = torch.tensor(self.init_alpha)

        pre_init_alpha = simpl_math.inverse_softplus(self.init_alpha)

        if self.auto_alpha:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=self.alpha_lr)
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.n_step = 0
        self.init_grad_clip_step = 0
        self.init_grad_clip = 5.0

        self.sample_time_logger = AverageMeter()

        try:
            self.save_prev_module()
            self.check_delta = True
        except:
            self.check_delta = False


    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
    
    def getPrior(self, states):
        with torch.no_grad():
            prior_dists = self.skill_prior.dist(self.policy.encode(states, prior = True))
        return prior_dists 

        
    def entropy(self, states, policy_dists, kl_clip = None):
        # with torch.no_grad():
        #     prior_dists = self.skill_prior.dist(self.policy.encode(states, prior = True))

        prior_dists = self.getPrior(states)

        if kl_clip is not None:                
            entropy = simpl_math.clipped_kl(policy_dists, prior_dists, clip = kl_clip)
        else:
            entropy = torch_dist.kl_divergence(policy_dists, prior_dists)

        return entropy, prior_dists 
    
    def update(self, step_inputs):
        self.train()

        # batch = self.buffer.sample(self.rl_batch_size)
        # self.episode = step_inputs['episode']

        batch = self.buffer.sample(self.rl_batch_size)
        # batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).to(self.device)
        self.episode = step_inputs['episode']
        self.n_step += 1

        self.stat = edict()

        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        if self.consistency_update:
            self.update_consistency(batch)

        self.update_qs(batch)

        if self.model_update:
            self.update_networks(batch)
        # ------------------- Alpha ------------------- # 
        self.update_alpha()

        if self.check_delta:
            self.calc_update_ratio("subgoal_generator")
            self.calc_update_ratio("inverse_dynamics")
            self.calc_update_ratio("dynamics")

        return self.stat
    

    def update_networks(self, batch):
        results = {}
        dist_out = self.policy.dist(batch, mode = "policy")
        policy_skill_dist = dist_out.policy_skill # 
        policy_skill = policy_skill_dist.rsample() 

        entropy_term, prior_dists = self.entropy(  batch.states,  policy_skill_dist, kl_clip= None) # policy의 dist로는 gradient 전파함 .
        
        # encoded_states = self.policy.encode(batch.states)
        # min_qs = torch.min(*[qf(encoded_states, policy_skill) for qf in self.qfs])

        # q_input = torch.cat((batch.states, batch.G, policy_skill), dim = -1)
        if self.gc:
            q_input = torch.cat((self.policy.encode(batch.states), batch.G, policy_skill), dim = -1)
        else:
            q_input = torch.cat((self.policy.encode(batch.states), policy_skill), dim = -1)

        min_qs = torch.min(*[qf( q_input ).squeeze(-1) for qf in self.qfs])
        policy_loss = (- min_qs + self.alpha * entropy_term).mean()
        policy_loss += self.aggregate_values(dist_out.additional_losses)


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results['policy_loss'] = policy_loss.item()
        results['kl'] = entropy_term.mean().item() # if self.prior_policy is not None else - entropy_term.mean()
        results['mean_policy_scale'], results['mean_prior_scale'] = get_scales(policy_skill_dist), get_scales(prior_dists)
        results['Q-value'] = min_qs.mean(0).item()

        # for k, v in dist_out['additional_losses'].items():
        #     results[k] = v.item()

        self.stat.update(results)
        self.stat.update(dist_out.additional_losses)


    @torch.no_grad()
    def compute_target_q(self, batch):
        batch_next = edict({**batch})
        batch_next['states'] = batch['next_states'].clone()

        policy_skill_dist = self.policy.dist(batch_next, mode = "policy").policy_skill
        policy_skill = policy_skill_dist.sample()

        # calculate entropy term
        entropy_term, prior_dists = self.entropy( batch_next.states, policy_skill_dist , kl_clip= 20) 
        if self.gc:
            q_input = torch.cat((self.policy.encode(batch_next.states), batch_next.G, policy_skill), dim = -1)
        else:
            q_input = torch.cat((self.policy.encode(batch_next.states), policy_skill), dim = -1)

        min_qs = torch.min(*[target_qf(q_input).squeeze(-1) for target_qf in self.target_qfs])
        soft_qs = min_qs - self.alpha*entropy_term

        rwd_term = batch.rewards
        ent_term = (1 - batch.dones) * self.discount * soft_qs

        return rwd_term, ent_term, entropy_term


    def update_qs(self, batch):
        qf_losses = []
        rwd_term, ent_term, entropy_term = self.compute_target_q(batch)
        target_qs = rwd_term + ent_term


        # policy skills 
        policy_skill_dist = self.policy.dist(batch, mode = "policy").policy_skill
        policy_skill = policy_skill_dist.sample() 
        cql_log_prob = policy_skill_dist.log_prob(policy_skill)

        # random skills 
        # 후보 : random skills, skill prior
        # 일단 skill prior로 해볼까..  ? 
        _, prior_dist = self.entropy(batch.states, policy_skill_dist)
        repeatedObs = batch.states.repeat(10, dim = 0)

        # 
        # random_skill = prior_dist.sample().repeat(10, dim = 0)
        
        random_skill_dist = get_fixed_dist(policy_skill.repeat(10, dim = 0), tanh = self.tanh)
        random_skill = random_skill_dist.sample()
        
        random_density = torch.log(0.5 ** self.cfg.skill_dim)


        if self.gc:
            q_input = torch.cat((self.policy.encode(batch.states), batch.G, batch.actions), dim = -1)
            q_input_cql = torch.cat((self.policy.encode(batch.states), batch.G, policy_skill), dim = -1)
            q_input_random = torch.cat((self.policy.encode(repeatedObs), batch.G.repeat(10, dim = 0), random_skill), dim = -1)

        else:
            q_input = torch.cat((self.policy.encode(batch.states), batch.actions), dim = -1)
            q_input_cql = torch.cat((self.policy.encode(batch.states), batch.G, policy_skill), dim = -1)
            q_input_random = torch.cat((self.policy.encode(batch.states), batch.G.repeat(10, dim = 0), random_skill), dim = -1)

        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            # qs = qf(self.q_inputs(batch)).squeeze(-1)
            # q_input = torch.cat((batch.states, batch.G, batch.actions), dim = -1)
            current_q = qf(q_input).squeeze(-1)
            cql_q = qf(q_input_cql).squeeze(-1)
            random_q = qf(q_input_random).squeeze(-1)

            
            cq = cql_q.repeat(10, dim = 0) - cql_log_prob
            rq = random_q - random_density

            conservative_loss = torch.logsumexp(torch.cat((cq, rq), dim = -1)).mean() - current_q.mean()

            # len(cql_q)는 원래 qf의 개수임. 
            # 왜 나누나요? 개별로 업데이트하니까 나눠줄 필요가 없음. 
            # conservative_loss = (conservative_weight * ((conservative_loss) / len(self.qfs)) - lagrange_thresh)


            conservative_loss = (self.conservative_weight * conservative_loss - self.lagrange_thresh)

            qs = qf(q_input).squeeze(-1)
            # 이 알파가 sac의 알파인지? 
            qf_loss = (qs - target_qs).pow(2).mean() + self.alpha * conservative_loss


            qf_optim.zero_grad()
            qf_loss.backward()
            qf_optim.step()
            qf_losses.append(qf_loss)


        update_moving_average(self.target_qfs, self.qfs, self.tau)

        results = {}
        results['qf_loss'] = torch.stack(qf_losses).mean()
        results['target_Q'] = target_qs.mean()
        results['rwd_term'] = rwd_term.mean()
        results['entropy_term'] = ent_term.mean()

        self.stat.update(results)



    def update_alpha(self):
        
        self.stat['target_kl'] = self.target_kl

        results = {}
        if self.auto_alpha:
            # dual gradient decent 
            alpha_loss = (self.alpha * (self.target_kl - self.stat.kl)).mean()

            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            results['alpha_loss'] = alpha_loss
        results['alpha'] = self.alpha

        self.stat.update(results)


    def warmup_Q(self, step_inputs):
        # self.train()
        self.stat = {}
        self.episode = step_inputs['episode']

        for _ in range(int(self.q_warmup)):
            batch = self.buffer.sample(self.rl_batch_size)
            # print(batch.G[:10, :2])

            # batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).to(self.device)

            if self.consistency_update:
                self.update_consistency(batch)
            self.update_qs(batch)
            # self.update_networks(batch)
        
        # # orig : 200 
        # for _ in range(int(self.q_warmup)):
        #     self.update(step_inputs)
        #     # batch = self.buffer.sample(self.rl_batch_size)
        #     # self.episode = step_inputs['episode']
        #     # self.n_step += 1
        #     # self.update_networks(batch)
        #     # ------------------- Alpha ------------------- # 



    
    def update_consistency(self, batch):

        consistency_losses = self.policy.dist(batch, mode = "consistency")
        consistency_loss = self.aggregate_values(consistency_losses)

        for module_name, optimizer in self.consistency_optims.items():
            optimizer['optimizer'].zero_grad()
            
        consistency_loss.backward()

        for module_name, optimizer in self.consistency_optims.items():
            self.grad_clip(optimizer['optimizer'])
            optimizer['optimizer'].step()

        # self.policy.soft_update() # 

        
        for module_name, meter in self.consistency_meters.items():
            target_metric = self.schedulers_metric[module_name]
            if target_metric is not None:
                meter.update(consistency_losses[target_metric], batch.states.shape[0])


        if (self.n_step + 1) % 256 == 0:
            for module_name, scheduler in self.consistency_schedulers.items():
                target_metric = self.schedulers_metric[module_name]
                meter = self.consistency_meters[module_name]
                if target_metric is not None:
                    scheduler.step(meter.avg)
                    meter.reset()

        self.stat.update(consistency_losses)

        # dist_out = self.policy.dist(batch, mode = "policy")
        # policy_skill_dist = dist_out.policy_skill # 
        # policy_skill = policy_skill_dist.rsample() 

        # entropy_term, prior_dists = self.entropy(  batch.states,  policy_skill_dist, kl_clip= None) # policy의 dist로는 gradient 전파함 .
        

        # if self.gc:
        #     q_input = torch.cat((self.policy.encode(batch.states), batch.G, policy_skill), dim = -1)
        # else:
        #     q_input = torch.cat((self.policy.encode(batch.states), policy_skill), dim = -1)
        # min_qs = torch.min(*[qf( q_input ).squeeze(-1) for qf in self.qfs])
        # policy_loss = (- min_qs + self.alpha * entropy_term).mean()
        # policy_loss += self.aggregate_values(dist_out.additional_losses)



    

    def q_inputs(self, batch, actions = None):
        encoded_states = self.policy.encode(batch.states)
        if actions is None:
            return torch.cat((encoded_states, batch.G, batch.actions), dim = -1)
        else:
            return torch.cat((encoded_states, batch.G, actions), dim = -1)
    
    @staticmethod
    def aggregate_values(loss_dict):
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss
    
    @torch.no_grad()
    def save_prev_module(self, target = "all"):
        self.prev_subgoal_generator = copy.deepcopy(self.policy.subgoal_generator).requires_grad_(False)
        self.prev_inverse_dynamics = copy.deepcopy(self.policy.inverse_dynamics).requires_grad_(False)
        self.prev_dynamics = copy.deepcopy(self.policy.dynamics).requires_grad_(False)

    @torch.no_grad()
    def calc_update_ratio(self, module_name):
        assert module_name in ['subgoal_generator', 'inverse_dynamics', 'dynamics'], "Invalid module name"

        prev_module = getattr(self, f"prev_{module_name}")
        module = getattr(self.policy, module_name)
        update_rate = []
        for (prev_p, p) in zip(prev_module.parameters(), module.parameters()):

            delta = (p - prev_p).norm()
            prev_norm =  prev_p.norm()
            
            if prev_norm.item() != 0:
                update_rate.append((delta / prev_norm).item())

        self.stat[f'delta_{module_name}'] = np.mean(update_rate)
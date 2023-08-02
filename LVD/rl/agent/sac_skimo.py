
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...modules import BaseModule
from ...contrib.simpl.math import clipped_kl, inverse_softplus
from ...utils import prep_state, kl_annealing, AverageMeter
from ...contrib import *
# from ...contrib.dists import *
from torch.optim import Adam
from easydict import EasyDict as edict


from ...utils import *
from ...models import BaseModel
from ...contrib import *

class SAC_Skimo(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.cfg=  config
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        rl_params = self.policy.get_rl_params()

        self.policy_optim = Adam(
            rl_params['policy'],
            # lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )


        # finetune : state_encoder, dynamics, reward function 
        if rl_params['consistency']:
            self.consistency_optims = {  name : {  
                'optimizer' : Adam( args['params'], lr = args['lr'] ),
                'metric' : args['metric']

            }   for name, args in rl_params['consistency'].items() }


            # scheduler_params = edict(
            #     factor = 0.5,
            #     patience = 10, # 4 epsisode
            #     verbose= True,
            # )


            # self.consistency_schedulers = {
            #     k : Scheduler_Helper(v['optimizer'], **scheduler_params, module_name = k) for k, v in self.consistency_optims.items()
            # }

            # self.schedulers_metric = {
            #     k : v['metric'] for k, v in self.consistency_optims.items()
            # }

            # self.consistency_meters = { k : AverageMeter() for k in self.consistency_optims.keys()}


        self.consistency_optims['Q'] = {
            'optimizer' : Adam( [
                {'params' : qf.parameters()} for qf in self.qfs
            ], lr = self.qf_lr ),
            'metric' : None     
        }
        
        



        # Alpha
        if isinstance(self.init_alpha, float):
            self.init_alpha = torch.tensor(self.init_alpha)

        pre_init_alpha = inverse_softplus(self.init_alpha)

        if self.auto_alpha:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=self.alpha_lr)
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.n_step = 0
        self.init_grad_clip_step = 0
        self.init_grad_clip = 5.0

        self.rho = 0.5

        # self.save_prev_module()


        self.sample_time_logger = AverageMeter()


    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def entropy(self, states, policy_dists, kl_clip = None):
        """
        Monte-Carlo estimate of KL-divergence 
        """
        
        with torch.no_grad():
            if self.cfg.env_name == "kitchen":
                prior_dists = self.skill_prior.dist(states[:, :self.cfg.n_pos])
            else:
                prior_dists = self.skill_prior.dist(states)


        skill_normal, skill = policy_dists.rsample_with_pre_tanh_value()
        log_prob_p = policy_dists.log_prob(skill, skill_normal)
        log_prob_q = prior_dists.log_prob(skill, skill_normal)
        entropy = log_prob_p.mean(0) - log_prob_q.mean(0)
        
        return entropy, prior_dists

        # with torch.no_grad():
        #     prior_dists = self.skill_prior.dist(self.policy.encode(states, prior = True))

        # if kl_clip is not None:                
        #     entropy = simpl_math.clipped_kl(policy_dists, prior_dists, clip = kl_clip)
        # else:
        #     entropy = torch_dist.kl_divergence(policy_dists, prior_dists)

        # return entropy, prior_dists 

    def update(self, step_inputs):
        self.train()

        # batch = self.buffer.sample(self.rl_batch_size)
        # self.episode = step_inputs['episode']

        batch = self.buffer.sample(self.rl_batch_size)
        # batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).to(self.device)
        # self.episode = step_inputs['episode']
        self.n_step += 1

        self.stat = edict()

        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #

        self.update_models(batch)
        self.update_policy(batch)

        # ------------------- Alpha ------------------- # 
        # self.update_alpha()
        self.policy.qfs = copy.deepcopy(self.qfs)

        return self.stat

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

        rwd_term = batch_next.rewards
        ent_term = (1 - batch_next.dones) * self.discount * soft_qs

        return rwd_term, ent_term, entropy_term


    def update_alpha(self, kl):
        results = {}
        if self.auto_alpha is True:
            # dual gradient decent 
            alpha_loss = (self.alpha * (self.target_kl - kl)).mean()

            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            results['alpha_loss'] = alpha_loss
            results['alpha'] = self.alpha

        return results
    


    def update_models(self, batch):
        """
        Trains 
        - Skill Dynamics
        - Reward function 
        - Q
        """
        
        # 1. skill seq을 순회하면서
        # 2. skill을 수행해 다음을 얻음 
            # a. h_next_pred
            # b. reward_pred 
        # 3. h_now, skill로 Q 예측
        # 4. loss 계산
            # a. state consistency 
            # b. reward loss 
            # c. value loss 
        
        # paper equation에 따르면 transition 단위로만 학습해도 됨. 
        
        consistency_loss = 0
        value_loss = 0

        T = batch.states.shape[1]
        
        state_pred = self.policy.encode(batch.states[:, 0], keep_grad= True)
        states_pred = []
        for t in range(T):
            states_pred.append(state_pred)

            rho = self.rho ** t

            now_batch = edict({ k : v[:, t] for k, v in batch.items()})
            
            # next state는 실제 값을 사용. 
            # state는 이전 step에서 예측된 값을 사용 
            now_batch['states'] = state_pred
            now_consistency_losses = self.policy.dist(now_batch, mode = "consistency")
            state_pred = now_consistency_losses.pop('htH_hat')
            consistency_loss += self.aggregate_values(now_consistency_losses) * rho

            if t == 0:
                self.stat['state_consistency'] = now_consistency_losses['state_consistency'].item() * 0.5
                self.stat['reward_loss'] = now_consistency_losses['reward_loss'].item() * 2

            # target Q : 실제 state, policy action을 사용해 value를 예측. 
            targetQ_batch = edict({ k : v[:, t] for k, v in batch.items()})
            rwd_term, ent_term, entropy_term = self.compute_target_q(targetQ_batch)
            target_qs = rwd_term + ent_term
            
            # Q : 예측된 state, 실제 action을 사용해 value를 예측. 
            qf_loss = 0
            if self.gc:
                q_input = torch.cat((self.policy.encode(now_batch.states), now_batch.G, now_batch.actions), dim = -1)
            else:
                q_input = torch.cat((self.policy.encode(now_batch.states), now_batch.actions), dim = -1)

            for qf in self.qfs:
                qs = qf(q_input).squeeze(-1)
                qf_loss += (qs - target_qs).pow(2).mean()
            
            value_loss += rho * qf_loss
        
            # # Additional reward prediction loss.
            # real_state = self.policy.encode(batch.states[:, t])
            # reward_pred = self.policy.reward_function(torch.cat((real_state, batch.actions[:, t]), dim = -1)).squeeze(-1) 

            # reward_loss += F.mse_loss(reward_pred, batch.rewards[:, t])
            # # Additional value prediction loss.
            # obs = hl_feat[t] if not cfg.sac else ob[t]["ob"]
            # q_pred = hl_agent.model.critic(obs, ac[t])
            # value_loss += mse(q_pred[0], q_target) + mse(q_pred[1], q_target)
        
        model_loss = consistency_loss + value_loss
        model_loss.register_hook(lambda grad: grad * (1 / T))

        for module_name, optimizer in self.consistency_optims.items():
            optimizer['optimizer'].zero_grad()

        model_loss.backward()

        for module_name, optimizer in self.consistency_optims.items():
            self.grad_clip(optimizer['optimizer'])
            optimizer['optimizer'].step()

        self.policy.soft_update() # 
        update_moving_average(self.target_qfs, self.qfs, self.tau)

        # results = {}
        # results['qf_loss'] = torch.stack(qf_losses).mean()
        # results['target_Q'] = target_qs.mean()
        # results['rwd_term'] = rwd_term.mean()
        # results['entropy_term'] = ent_term.mean()
        # self.stat.update(results)
        # self.stat.update(consistency_losses)

        self.states_pred = torch.stack(states_pred, dim = 1)#.clone().detach()

    def update_policy(self, batch):

        # policy loss 
        # rollout_high_states = step_inputs['rollout_high_states']

		# Loss is a weighted sum of Q-values
        policy_loss = 0

    
        states = batch.states 
        T = states.shape[1]

        # 1. 실제 state로 prior 계산
        # 2. 예측한 state로 
        
        for t in range(T):
            rho = self.rho ** t
            # state = batch.states[:, t]
            state, G = self.states_pred[:, t], batch.G[:, t]

            policy_inputs = edict(
                # states, G
                states = state, # ctor loss는 rollout state로 구한다.  
                G = G
            )
        
            policy_dists = self.policy.dist(policy_inputs, mode = "policy")['policy_skill'] # prior policy mode.
            policy_skill = policy_dists.rsample() 
            entropy_term, prior_dist = self.entropy(batch.states[:, t], policy_dists, kl_clip= False) # policy의 dist로는 gradient 전파함 .
            
            
            if self.gc:
                q_input = torch.cat((state, G, policy_skill), dim = -1)
            else:
                q_input = torch.cat((state, policy_skill), dim = -1)

            min_qs = torch.min(*[qf(q_input) for qf in self.qfs])
            ent_loss = self.alpha * entropy_term

            # q_values += min_qs.clone().detach().mean(0).item()
            # entropy_terms += entropy_term.clone().detach().mean().item() 
            policy_loss += ((- min_qs + ent_loss) * rho).mean() 
            
        policy_loss.register_hook(lambda grad: grad * (1 / T))

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results = {}
    

        results['policy_loss'] = policy_loss.item()
        # results['kl'] = entropy_terms # if self.prior_policy is not None else - entropy_term.mean()
        # results['Q-value'] = q_values
        
        return results
        

    def warmup_Q(self, step_inputs):
        # self.train()
        self.stat = {}
        self.episode = step_inputs['episode']

        for _ in range(self.q_warmup):
            batch = self.buffer.sample(self.rl_batch_size)
            q_results = self.update_models(batch)
        
        # # orig : 200 
        for _ in range(int(self.q_warmup)):
            batch = self.buffer.sample(self.rl_batch_size)
            self.update(batch)
            # batch = self.buffer.sample(self.rl_batch_size)
            # self.episode = step_inputs['episode']
            # self.n_step += 1
            # self.update_networks(batch)
            # ------------------- Alpha ------------------- # 

        # for k, v in q_results.items():
        #     stat[k] = v 

        self.policy.qfs = copy.deepcopy(self.qfs)

    
    def next_batch(self, batch, t):
        """
        batch내의 모든 값이 sequence 형태.
        딱 1 time step 만 줘야 함. 
        
        """
        
        # batch_next = edict({**batch})

        batch_next = edict({ k : v[:, t]  for k, v in batch.items()})
        batch_next['states'] = batch['next_states'].clone()


        return batch_next
    
    @staticmethod
    def aggregate_values(loss_dict):
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss
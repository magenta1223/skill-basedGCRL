
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from ..utils import *
from ..models import BaseModel
from ..contrib import *


class SAC(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        
        sac_params = self.policy.get_rl_params()

        self.policy_optim = torch.optim.Adam(
            sac_params['policy'],
            lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )

        self.consistency_optim = torch.optim.Adam(
            sac_params['consistency'],
            lr = self.consistency_lr # 낮추면 잘 안됨. 왜? 
        )

                
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


    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def entropy(self, states, policy_dists, kl_clip = None):
        with torch.no_grad():
            # hidden state를 쓸 수도 있고, 아닐수도 있음. 
            encoded_states = self.policy.encode(states)
            prior_dists = self.skill_prior.dist(encoded_states)

        if kl_clip is not None:                
            entropy = simpl_math.clipped_kl(policy_dists, prior_dists, clip = kl_clip)
        else:
            entropy = torch_dist.kl_divergence(policy_dists, prior_dists)

        return entropy, prior_dists 
    
    def update(self, step_inputs):
        self.train()

        batch = self.buffer.sample(self.rl_batch_size)
        self.episode = step_inputs['episode']

        self.n_step += 1


        self.stat = edict()

        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        if self.consistency_update:
            self.update_consistency(batch)

        self.update_qs(batch)
        self.update_networks(batch)

        # for k, v in q_results.items():
        #     stat[k] = v         
        # for k, v in policy_results.items():
        #     stat[k] = v 


        # ------------------- Alpha ------------------- # 
        self.update_alpha()

        #     self.save_prev_module()
        return self.stat
    

    def update_networks(self, batch):
        results = {}
        dist_out = self.policy.dist(batch)
        policy_skill_dist = dist_out.policy_skill # 
        policy_skill = policy_skill_dist.rsample() 

        entropy_term, prior_dists = self.entropy(  batch.states,  policy_skill_dist, kl_clip= None) # policy의 dist로는 gradient 전파함 .
        
        # encoded_states = self.policy.encode(batch.states)
        # min_qs = torch.min(*[qf(encoded_states, policy_skill) for qf in self.qfs])
        min_qs = torch.min(*[qf( self.q_inputs(batch, policy_skill) ) for qf in self.qfs])

        policy_loss = (- min_qs + self.alpha * entropy_term).mean()

        # policy_loss += torch.sum(dist_out.additional_losses.values())
        policy_loss += torch.sum(torch.tensor(list(dist_out.additional_losses.values())))


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
        
        batch_next_states = edict({**batch})
        batch_next_states['states'] = batch['next_states']
        
        policy_skill_dist = self.policy.dist(batch_next_states).policy_skill
        policy_skill = policy_skill_dist.sample()

        # calculate entropy term
        entropy_term, prior_dists = self.entropy( batch_next_states.states, policy_skill_dist , kl_clip= 20) 
        min_qs = torch.min(*[target_qf(  self.q_inputs(batch_next_states, policy_skill)  ) for target_qf in self.target_qfs])

        soft_qs = (- min_qs + self.alpha * entropy_term)

        rwd_term = batch.rewards
        ent_term = (1 - batch.dones) * self.discount * soft_qs

        return rwd_term, ent_term, entropy_term


    def update_qs(self, batch):
        rwd_term, ent_term, entropy_term = self.compute_target_q(batch)
        target_qs = rwd_term + ent_term

        qf_losses = []  
        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            qs = qf(self.q_inputs(batch))
            qf_loss = (qs - target_qs).pow(2).mean()
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
        batch = self.buffer.sample(self.rl_batch_size)
        self.episode = step_inputs['episode']



        for _ in range(self.q_warmup):
            if self.consistency_update:
                self.update_consistency(batch)
            self.update_qs(batch)
                        

    
    def update_consistency(self, batch):

        results = {}
        consistency_losses = self.policy.dist(batch)['consistency_losses']

        consistency_loss = torch.sum(consistency_losses.values())

        self.consistency_optim.zero_grad()
        consistency_loss.backward()
        self.grad_clip(self.policy_optim)
        self.consistency_optim.step()
        self.policy.soft_update()

        # # metrics
        # with torch.no_grad():
        #     dist = get_dist(step_inputs['loc'], step_inputs['scale'].log(), tanh = self.tanh)
        #     iD_kld = torch_dist.kl_divergence(dist, outputs['inverse_D']).mean() # KL (post || prior)
        # self.consistency_meter.update(iD_kld.item(), step_inputs['batch_size'])

        # # scheduler step
        # if (self.n_step + 1) % 256 == 0:
        #     self.consistency_scheduler.step(self.consistency_meter.avg)
        #     self.consistency_meter.reset()
        
        # log 

        self.stat.update(consistency_losses)
    

    def q_inputs(self, batch, actions = None):
        encoded_states = self.policy.encode(batch.states)
        if actions is None:
            return torch.cat((encoded_states, batch.G, batch.actions), dim = -1)
        else:
            return torch.cat((encoded_states, batch.G, actions), dim = -1)

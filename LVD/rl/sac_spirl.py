
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..modules.base import BaseModule
from ..contrib.simpl.math import clipped_kl, inverse_softplus
from ..utils import prep_state, nll_dist, kl_annealing, get_dist, AverageMeter, Scheduler_Helper
from ..contrib.momentum_encode import update_moving_average
from ..contrib.dists import *

from easydict import EasyDict as edict
import datetime

class SAC(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        # Finetune
        rl_params = self.policy.get_rl_params()
        self.policy_optim = torch.optim.Adam(
            rl_params['policy'],
            lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )
        
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

        self.sample_time_logger = AverageMeter()

    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def entropy(self, batch, policy_dist, kl_clip = None):
        with torch.no_grad():
            prior_dists = self.skill_prior.dist(batch.states)

        if kl_clip is not None:                
            entropy = clipped_kl(policy_dist, prior_dists, clip = kl_clip)
        else:
            entropy = torch_dist.kl_divergence(policy_dist, prior_dists)
        return entropy, prior_dists 
    
    def grad_clip(self, optimizer):
        if self.n_step < self.init_grad_clip_step:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.init_grad_clip) 

    def update(self, step_inputs):
        self.train()
        # self.policy.inverse_dynamics.state_encoder.eval()

        batch = self.buffer.sample(self.rl_batch_size)
        self.episode = step_inputs['episode']
        batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).cuda()
        self.n_step += 1

        stat = {}
        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        q_results = self.update_qs(batch)
        policy_results = self.update_networks(batch)

        for k, v in q_results.items():
            stat[k] = v 
        
        for k, v in policy_results.items():
            stat[k] = v 
        # ------------------- Logging ------------------- # 
        stat['target_kl'] = self.target_kl

        # ------------------- Alpha ------------------- # 
        alpha_results = self.update_alpha(stat['kl'])
        for k, v in alpha_results.items():
            stat[k] = v 

        return stat
    
    def update_networks(self, batch):
        results = {}
        dist_out = self.policy.dist(batch)
        policy_skill_dist = dist_out['policy_skill'] # 
        policy_skill = policy_skill_dist.rsample() 

        entropy_term, prior_dists = self.entropy(batch, policy_skill_dist, kl_clip= None) # policy의 dist로는 gradient 전파함 .
        q_input = torch.cat((batch.states, policy_skill), dim = -1)
        min_qs = torch.min(*[qf(q_input).squeeze(-1) for qf in self.qfs])

        # min_qs = torch.min(*[qf(step_inputs['states'], step_inputs['policy_actions']) for qf in self.qfs])
        policy_loss = (- min_qs + self.alpha * entropy_term).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results['policy_loss'] = policy_loss.item()
        results['kl'] = entropy_term.mean().item() # if self.prior_policy is not None else - entropy_term.mean()
        if self.tanh:
            results['mean_policy_scale'] = policy_skill_dist._normal.base_dist.scale.abs().mean().item() 
            results['mean_prior_scale'] = prior_dists._normal.base_dist.scale.abs().mean().item()
        else:
            results['mean_policy_scale'] = policy_skill_dist.base_dist.scale.abs().mean().item() 
            results['mean_prior_scale'] = prior_dists.base_dist.scale.abs().mean().item()

        results['Q-value'] = min_qs.mean(0).item()

        return results

    @torch.no_grad()
    def compute_target_q(self, batch):
        batch_next = edict({**batch})
        batch_next['states'] = batch['next_states']

        policy_skill_dist = self.policy.dist(batch_next)['policy_skill']
        actions = policy_skill_dist.sample()

        # calculate entropy term
        entropy_term, prior_dists = self.entropy(batch_next, policy_skill_dist, kl_clip= 20) # policy의 dist로는 gradient 전파함 .

        q_input = torch.cat((batch_next.states, actions), dim = -1)
        min_qs = torch.min(*[target_qf(q_input).squeeze(-1) for target_qf in self.target_qfs])
        soft_qs = min_qs - self.alpha*entropy_term

        rwd_term = batch['rewards'].cuda()
        ent_term = (1 - batch['dones'].cuda())*self.discount*soft_qs

        return rwd_term, ent_term, entropy_term

    def update_qs(self, batch):
        rwd_term, ent_term, entropy_term = self.compute_target_q(batch)
        target_qs = rwd_term + ent_term

        # Q-function의 parameter 업데이트 양을 조사해보자. 
        # 퍼센티지로? ㄴㄴ 크기도 아닌데 
                
        qf_losses = []  
        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            q_input = torch.cat((batch['states'], batch['actions']), dim = -1)
            qs = qf(q_input).squeeze(-1)
            # qs = qf(step_inputs['states'], step_inputs['actions'])
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

        return results



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


    def warmup_Q(self, step_inputs):
        # self.train()

        batch = self.buffer.sample(self.rl_batch_size)
        self.episode = step_inputs['episode']
        batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).cuda()

        stat = {}

        for _ in range(self.q_warmup):
            q_results = self.update_qs(batch)
                        
        for k, v in q_results.items():
            stat[k] = v 

        return stat
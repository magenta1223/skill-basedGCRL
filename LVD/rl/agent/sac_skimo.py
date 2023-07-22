
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

class SAC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
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
        with torch.no_grad():
            prior_dists = self.skill_prior.dist(self.policy.encode(states, prior = True))

        if kl_clip is not None:                
            entropy = simpl_math.clipped_kl(policy_dists, prior_dists, clip = kl_clip)
        else:
            entropy = torch_dist.kl_divergence(policy_dists, prior_dists)

        return entropy, prior_dists 

    # def entropy(self, inputs, kl_clip = False):
    #     inputs = {**inputs}

    #     with torch.no_grad():
    #         prior_dist = self.policy.prior_policy(inputs, "prior")['prior']
    #     if kl_clip:                
    #         entropy = clipped_kl(inputs['dist'], prior_dist, clip = self.kl_clip)
    #     else:
    #         entropy = torch_dist.kl_divergence(inputs['dist'], prior_dist)
    #     return entropy, prior_dist 
    

    
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



    @torch.no_grad()
    def compute_target_q(self, batch_next):
        # batch_next = edict({**batch})
        # batch_next['states'] = batch['next_states'].clone()

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
    
    def update_Q_models(self, batch):


        N, T = batch.states.shape[:2]

        # high_states = self.policy.prior_policy.state_encoder(states.view(N * T, -1)).view(N, T, -1)
        # high_next_states = self.policy.prior_policy.state_encoder(next_states.view(N * T, -1)).view(N, T, -1)
        
        # encode 할 때 3차원이면 flat, unflat하는 기능 추가 
        hts = self.policy.encode(batch.states)
        hts_next = self.policy.encode(batch.next_states)

        # skills = step_inputs['actions']
        skills = batch.actions
        
        # high_state = high_states[:, 0]
        ht = hts[:, 0]
        consistency_loss, reward_loss, value_loss = 0, 0, 0
        rollout_hts = []

        for t in range(T):
            # ---------- value prediction ---------- #
            # Q-value
            if self.gc:
                q_input = torch.cat((ht, batch.G, batch.actions[:, t]), dim = -1)
            else:
                q_input = torch.cat((ht, batch.actions[:, t]), dim = -1)

            # q1, q2 = [qf(high_state, step_inputs['actions'][:, t]) for qf in self.qfs]
            q1, q2 = [qf(q_input) for qf in self.qfs]
            

            # target_q_inputs = dict(
            #     next_states = next_states[:, t],
            #     G = G,
            #     q_next_states = high_next_states[:, t],
            #     rewards = rewards[:, t],
            #     dones = dones[:, t]
            # )
            
            next_batch = self.next_batch(batch, t)

            # target Q
            with torch.no_grad():
                rwd_term, ent_term, entropy_term = self.compute_target_q(next_batch)
                target_qs = rwd_term + ent_term


            # ---------- value function, state consistency ---------- #
            rollout_inputs = dict(
                states = ht,
                actions = skills[:, t]
            )
            outputs = self.policy.rollout_latent(rollout_inputs)

            # next_state_pred
            # high_next_state, rewards_pred = outputs['next_states'], outputs['rewards_pred']
            ht_next_pred, r_pred = outputs.next_states, outputs.rewards_pred
        
            # discounted 
            rho = (self.rho ** t)
            consistency_loss += rho * F.mse_loss(ht_next_pred, hts_next[:, t])
            reward_loss += rho * F.mse_loss(r_pred, batch.rewards[:, t])
            value_loss += rho * (F.mse_loss(q1, target_qs) + F.mse_loss(q2, target_qs))


            rollout_hts.append(high_state.clone().detach())
            high_state = high_next_state


        total_loss = (consistency_loss * 2 + reward_loss * 0.5 + value_loss * 0.1) / T

        for qf_optim in self.qf_optims:
            qf_optim.zero_grad()
        self.others_optim.zero_grad()

        total_loss.backward()

        for qf_optim in self.qf_optims:
            qf_optim.step()
        self.others_optim.step()

        # qf_losses = [value_loss]  

        
        
        return rollout_high_states


    def update_policy(self, step_inputs):

        # policy loss 
        rollout_high_states = step_inputs['rollout_high_states']
		# Loss is a weighted sum of Q-values
        policy_loss = 0
        q_values = 0
        entropy_terms = 0
    
        states = step_inputs['states']

        for t, high_state in enumerate(rollout_high_states):
            policy_inputs = dict(
                # states, G
                states = states[:, t], # skill prior는 GT state로 구하고
                high_state = high_state, # actor loss는 rollout state로 구한다. 
                G = step_inputs['G']
            )
        
            policy_inputs['dist'] = self.policy.dist(policy_inputs, latent = True)['policy_skill'] # prior policy mode.
            policy_inputs['policy_actions'] = policy_inputs['dist'].rsample() 
            entropy_term, prior_dist = self.entropy(policy_inputs, kl_clip= False) # policy의 dist로는 gradient 전파함 .
            min_qs = torch.min(*[qf(high_state, policy_inputs['policy_actions']) for qf in self.qfs]) * (self.rho ** t)
            ent_loss = self.alpha * entropy_term * (self.rho ** t)

            q_values += min_qs.clone().detach().mean(0).item()
            entropy_terms += entropy_term.clone().detach().mean().item() 


            policy_loss += (- min_qs + ent_loss).mean() 
            
        policy_loss /= len(rollout_high_states)
            

        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results = {}
    

        results['policy_loss'] = policy_loss.item()
        results['kl'] = entropy_terms # if self.prior_policy is not None else - entropy_term.mean()
        results['Q-value'] = q_values

        update_moving_average(self.target_qfs, self.qfs, self.tau)
        self.policy.prior_policy.soft_update()
        
        return results
        

    def warmup_Q(self, step_inputs):
        # self.train()
        # self.policy.inverse_dynamics..eval()

        batch = self.buffer.sample(step_inputs['batch_size']).to(self.device)
        self.episode = step_inputs['episode']

        with torch.no_grad():
            states = prep_state(batch.states, self.device)
            next_states = prep_state(batch.next_states, self.device)
            # step_inputs['batch'] = batch
            step_inputs['rewards'] = batch.rewards
            step_inputs['dones'] = batch.dones
            step_inputs['states'] = states
            step_inputs['next_states'] = next_states
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['batch_size'], 1).cuda()
            step_inputs['raw_states'] = states
            step_inputs['raw_next_states'] = next_states

            step_inputs['done'] = True

   
            step_inputs['actions'] = prep_state(batch.actions, self.device) # high-actions


        stat = {}

        for _ in range(self.q_warmup_steps):
            q_results = self.update_Q_models(step_inputs)
                        
        # for k, v in q_results.items():
        #     stat[k] = v 

        return stat
    
    def next_batch(self, batch, t):
        """
        batch내의 모든 값이 sequence 형태.
        딱 1 time step 만 줘야 함. 
        
        """
        
        # batch_next = edict({**batch})

        batch_next = edict({ k : v[:, t]  for k, v in batch.items()})
        batch_next['states'] = batch['next_states'].clone()


        return batch_next

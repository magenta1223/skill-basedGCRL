import torch
import copy

import math

from ...modules.base import BaseModule
from ...utils.utils import *
from ...contrib.momentum_encode import update_moving_average

class Skimo_Prior(BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        super().__init__(submodules)
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.step = 0

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # hard update 
        if self.step % 2 == 0:
            update_moving_average(self.target_state_encoder, self.state_encoder)

    def forward(self, inputs, *args, **kwargs):
        # print(self.methods.keys())
        """
        Jointly Optimize 
        - State Encoder / Decoder
        - Inverse Dynamcis
        - Dynamics
        - Subgoal Generator
        - Skill Encoder / Decoder (at upper level)
        """
        if self.training:
            self.step += 1

        states, skill, G = inputs['states'], inputs['skill'], inputs['G']
        N, T, _ = states.shape
        skill_length = T - 1 

        # -------------- State Enc / Dec -------------- #
        # jointly learn
        states_repr = self.state_encoder(states.view(N * T, -1))

        state_emb = states_repr.view(N, T, -1)[:, 0]
        states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda()
        states_hat = self.state_decoder(states_repr).view(N, T, -1)

            # G = self.state_encoder(G)
        hts = states_repr.view(N, T, -1).clone().detach()
        ht = hts[:, 0]

        with torch.no_grad():
            htH = self.target_state_encoder(states[:, -1])

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.prior_policy.dist(states[:, 0], detached = True)

        # ------------------ Skill Dynamics ------------------- #
        dynamics_input = torch.cat((ht, skill), dim = -1)
        D = self.dynamics(dynamics_input)

        # -------------- High-level policy -------------- #
        policy_skill =  self.highlevel_policy.dist(torch.cat((ht.clone().detach(), G), dim = -1))

        # -------------- Rollout for metric -------------- #
        # dense execution with loop (for metric)
        with torch.no_grad():
            subgoal_recon_D = self.state_decoder(D)


        result = {
            # states
            "states" : states,
            "states_repr" : state_emb,
            "hts" : hts,
            "states_hat" : states_hat,
            "states_fixed_dist" : states_fixed,
            # state conditioned prior
            "prior" : prior,
            "prior_detach" : prior_detach,

            # Ds
            "D" : D,
            "D_target" : htH, 

            # highlevel policy
            "policy_skill" : policy_skill,

            # for metric
            "z_invD" : skill,
            "subgoal_recon_D" : subgoal_recon_D,


        }


        return result
    
    def dist(self, inputs, latent = False):
        """
        latent : for backpropagation through time
        """

        if latent:
            ht, G = inputs['high_state'].detach(), inputs['G']
        else:
            state, G = inputs['states'], inputs['G']
            with torch.no_grad():
                ht = self.state_encoder(state)

        policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1))

        return dict(
            policy_skill = policy_skill
        )
    

    @torch.no_grad()
    def estimate_value(self, state, skills, G, horizon, qfs):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            # step을 추가하자
            state_skill = torch.cat((state, skills[:, t]), dim  = -1)
            state = self.dynamics(state_skill)
            reward = self.reward_function(state_skill) 
            value += discount * reward
            discount *= self.rl_discount
        
        # policy_skill = self.prior_policy(policy_inputs, "eval")['policy_skill'].sample()
        # if self.gc:
        #     policy_skill =  self.highlevel_policy.dist(torch.cat((state, G), dim = -1)).sample()
        # else:
        #     policy_skill =  self.highlevel_policy.dist(state).sample()

        policy_skill =  self.highlevel_policy.dist(torch.cat((state, G), dim = -1)).sample()
        q_values = [  qf( state, policy_skill).unsqueeze(-1)   for qf in qfs]
        value += discount * torch.min(*q_values) # 마지막엔 Q에 넣어서 value를 구함. 
        return value

    @torch.no_grad()
    def rollout(self, inputs):
        ht, G = inputs['states'], inputs['G']
        planning_horizon = inputs['planning_horizon']

        skills = []
        for i in range(planning_horizon):
            # if self.gc:
            #     policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1)).sample()
            # else:
            #     policy_skill =  self.highlevel_policy.dist(ht).sample()
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1)).sample()
            dynamics_input = torch.cat((ht, policy_skill), dim = -1)
            ht = self.dynamics(dynamics_input)
            skills.append(policy_skill)
        
        return dict(
            policy_skills = torch.stack(skills, dim=1)
        )
    
    @torch.no_grad()
    def cem_planning(self, inputs):
        """
        Cross Entropy Method
        """

        planning_horizon  = int(self._horizon_decay(self._step))

        states, qfs = inputs['states'], inputs['qfs']
        states = self.state_encoder(states)

        # Sample policy trajectories.        
        rollout_inputs = dict(
            states = states.repeat(self.num_policy_traj, 1),
            G = inputs['G'].repeat(self.num_policy_traj, 1) ,
            planning_horizon = planning_horizon
        )

        policy_skills =  self.rollout(rollout_inputs)['policy_skills']

        # CEM optimization.
        state_cem = states.repeat(self.num_policy_traj + self.num_sample_traj, 1)
        
        # zero mean, unit variance
        momentums = torch.zeros(self.num_sample_traj, planning_horizon, self.skill_dim * 2, device= self.device)
        loc, log_scale = momentums.chunk(2, -1)
        dist = get_dist(loc, log_scale= log_scale, tanh = self.tanh)

        for _ in range(self.cem_iter):
            # sampled skill + policy skill
            sampled_skill = dist.sample()
            skills = torch.cat([sampled_skill, policy_skills], dim=0)

            # reward + value 
            imagine_return = self.estimate_value(state_cem, skills, inputs['G'].repeat(skills.shape[0], 1) , planning_horizon, qfs)
                        
            # sort by reward + value
            elite_idxs = imagine_return.sort(dim=0)[1].squeeze(1)[-self.num_elites :]
            elite_value, elite_skills = imagine_return[elite_idxs], skills[elite_idxs]

            # Weighted aggregation of elite plans.
            score = torch.softmax(self.cem_temperature * elite_value, dim = 0).unsqueeze(-1)

            dist = self.score_weighted_skills(loc, score, elite_skills)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        skill = elite_skills[np.random.choice(np.arange(self.num_elites), p=score), 0] 
        return skill


    def finetune(self, inputs):
        """
        Finetune state encoder, dynamics
        """

        states, next_states, skill = inputs['states'], inputs['next_states'], inputs['actions']

        ht = self.state_encoder(states)
        htH = self.target_state_encoder(next_states)
        htH_hat = self.dynamics(torch.cat((ht, skill), dim = -1))
        
        rewards_pred = self.reward_function(torch.cat((ht, skill), dim = -1)) 

        
        result = {
            "ht" : ht,
            "subgoal_target" : htH,
            "subgoal" : htH_hat,
            "rwd_pred" : rewards_pred
        }

        return result
    
    def rollout_latent(self, inputs):

        ht, skills = inputs['states'], inputs['actions']
        dynamics_input = torch.cat((ht, skills), dim = -1)

        next_states = self.dynamics(dynamics_input)
        rewards_pred = self.reward_function(dynamics_input) 

        return dict(
            next_states = next_states,
            rewards_pred = rewards_pred
        )


    
    def _std_decay(self, step):
        # from rolf
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 0.5 * (1-mix) + 0.01 * mix

    def _horizon_decay(self, step):
        # from rolf
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 1 * (1-mix) + self.planning_horizon * mix


    def score_weighted_skills(self, loc, score, skills):
        weighted_loc = (score * skills).sum(dim=0)
        weighted_std = torch.sqrt(torch.sum(score * (skills - weighted_loc.unsqueeze(0)) ** 2, dim=0))
        
        # soft update 
        loc = self.cem_momentum * loc + (1 - self.cem_momentum) * weighted_loc
        log_scale = torch.clamp(weighted_std, self._std_decay(self._step), 2).log() # new_std의 최소값. .. 을 해야돼? 
        dist = get_dist(loc, log_scale, tanh = self.tanh)

        return dist
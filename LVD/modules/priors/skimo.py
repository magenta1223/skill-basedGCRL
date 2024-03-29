import numpy as np
import torch
import copy
from easydict import EasyDict as edict
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import update_moving_average
from ...contrib import TanhNormal
from torch.nn import functional as F

class SkiMo_Prior(ContextPolicyMixin, BaseModule):
    """
    """
    def __init__(self, **submodules):
        super().__init__(submodules)
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.step = 0
        self.qfs = None
        self.rollout_step = 0

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
        - High Policy
        - Skill Encoder / Decoder (at upper level)
        """
        if self.training:
            self.step += 1

        states, skill, G = inputs['states'], inputs['skill'], inputs['G']
        N, T, _ = states.shape
        skill_length = T - 1 

        # -------------- State Enc / Dec -------------- #
        # jointly learn
        states_repr = self.state_encoder(states.view(N * T, -1))[0]
        states_hat = self.state_decoder(states_repr).view(N, T, -1)[0]

            # G = self.state_encoder(G)
        hts = states_repr.view(N, T, -1).clone().detach()
        ht = hts[:, 0]

        with torch.no_grad():
            htH = self.target_state_encoder(states[:, -1])

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.forward_prior(states)
        
        # ------------------ Skill Dynamics ------------------- #
        dynamics_input = torch.cat((ht, skill), dim = -1)
        D = self.dynamics(dynamics_input)

        # -------------- High-level policy -------------- #
        policy_skill =  self.highlevel_policy.dist(torch.cat((ht.clone().detach(), G), dim = -1))

        # -------------- Rollout for metric -------------- #
        with torch.no_grad():
            subgoal_recon_D = self.state_decoder(D)


        result = edict({
            # states
            "states" : states,
            "hts" : hts,
            "states_hat" : states_hat,
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
        })


        return result

    def forward_prior(self, start):
        if len(start.shape) > 2:
            start = start[:, 0]
        
        if self.cfg.manipulation:
            start = start[:, :self.cfg.n_pos]
            
        return self.skill_prior.dist(start, detached = True)


    @torch.no_grad()
    def act(self, states, G):
        self.rollout_step += 1

        batch = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )
        
        if self.qfs is not None:
            dist = self.dist(batch, mode = "act").policy_skill
            if isinstance(dist, TanhNormal):
                if self.explore: 
                    z_normal, z = dist.sample_with_pre_tanh_value()
                    return to_skill_embedding(z_normal), to_skill_embedding(z)
                else:  
                    return to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2)), to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2))

            else:
                return None, to_skill_embedding(dist.sample())
        else:
            dist = self.dist(batch, mode = "act").policy_skill
            if isinstance(dist, TanhNormal):
                if self.explore: 
                    z_normal, z = dist.sample_with_pre_tanh_value()
                    return to_skill_embedding(z_normal), to_skill_embedding(z)
                else:  
                    return to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2)), to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2))

            else:
                return None, to_skill_embedding(dist.sample())
    
    def dist(self, batch, mode = "policy", latent = False):
        if mode == "consistency":
            return self.consistency(batch)
        elif mode == "policy":
            self.step += 1
            state, G = batch.states, batch.G
            if state.shape[-1] == self.cfg.latent_state_dim:
                ht = state 
            else:
                with torch.no_grad():
                    ht = self.state_encoder(state)
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1))
            return edict(
                policy_skill = policy_skill,
                additional_losses = {}
            )
        else: 
            # act 
            state, G = batch.states, batch.G
            with torch.no_grad():
                ht = self.state_encoder(state)
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1))
            return edict(
                policy_skill = policy_skill
            )
        
    @torch.no_grad()
    def estimate_value(self, state, skills, G, horizon, qfs):
        value, discount = 0, 1
        for t in range(horizon):
            next_state = self.dynamics( torch.cat((state, skills[:, t]), dim  = -1))
            reward = self.reward_function( torch.cat((state, G, skills[:, t]), dim  = -1)).squeeze(-1) 

            value += discount * reward
            discount *= self.cfg.discount

            state = next_state
        
        policy_skill =  self.highlevel_policy.dist(torch.cat((state, G), dim = -1)).sample()
        q_values = [  qf(torch.cat((state, G, policy_skill), dim = -1)).squeeze(-1)   for qf in qfs]
        
        value += discount * torch.min(*q_values) 
        return value

    @torch.no_grad()
    def rollout(self, batch):
        ht, G = batch.states, batch.G
        planning_horizon = batch.planning_horizon
        skills = []
        skills_normal = []
        for i in range(planning_horizon):
            policy_skill_normal, policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1)).sample_with_pre_tanh_value()
            skills.append(policy_skill)
            skills_normal.append(policy_skill_normal)
            dynamics_input = torch.cat((ht, skills[i]), dim = -1)
            ht = self.dynamics(dynamics_input)
        return edict(
            policy_skills = torch.stack(skills, dim=1),
            policy_skills_normal = torch.stack(skills_normal, dim=1)
        )
    @torch.no_grad()
    def cem_planning(self, batch):
        planning_horizon  = int(self._horizon_decay(self.rollout_step))        
        qfs = self.qfs 
        states = batch.states
        states = self.state_encoder(states)
        # Sample policy trajectories.        
        rollout_batch = edict(
            states = states.repeat(self.cfg.num_policy_traj, 1),
            G = batch.G.repeat(self.cfg.num_policy_traj, 1) ,
            planning_horizon = planning_horizon
        )

        rollouts =  self.rollout(rollout_batch)
        policy_skills, policy_skills_normal = rollouts['policy_skills'], rollouts['policy_skills_normal']

        # CEM optimization.
        state_cem = states.repeat(self.cfg.num_policy_traj + self.cfg.num_sample_traj, 1)
        
        # zero mean, unit variance
        momentums = torch.zeros(self.cfg.num_sample_traj, planning_horizon, self.cfg.skill_dim * 2, device= self.device)

        loc, log_scale = momentums.chunk(2, -1)
        dist = get_dist(loc, log_scale= log_scale, tanh = self.cfg.tanh)

        for _ in range(self.cfg.cem_iter):
            # sampled skill + policy skill
            sampled_skill_normal, sampled_skill = dist.sample_with_pre_tanh_value()

            skills_normal = torch.cat([sampled_skill_normal, policy_skills_normal], dim=0)
            skills = torch.cat([sampled_skill, policy_skills], dim=0)

            # reward + value 
            imagine_return = self.estimate_value(state_cem, skills, batch.G.repeat(skills.shape[0], 1) , planning_horizon, qfs)
                        
            # sort by reward + value
            elite_idxs = imagine_return.sort(dim=0)[1][-self.cfg.num_elites :]
            elite_value, elite_skills, elite_skills_normal = imagine_return[elite_idxs], skills[elite_idxs],  skills_normal[elite_idxs]

            # Weighted aggregation of elite plans.
            score = torch.softmax(self.cfg.cem_temperature * elite_value, dim = 0).unsqueeze(-1)

            dist, loc = self.score_weighted_skills(loc, score, elite_skills)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        sample_indices = np.random.choice(np.arange(self.cfg.num_elites), p=score)
        skill = elite_skills[sample_indices, 0] 
        skill_normal = elite_skills_normal[sample_indices, 0] 

        return skill_normal, skill

    def consistency(self, batch):
        """
        Finetune state encoder, dynamics
        """

        states, next_states, skill, G, relabeled_G = batch.states, batch.next_states, batch.actions, batch.G, batch.relabeled_G
        
        if states.shape[1] == self.cfg.latent_state_dim:
            ht = states 
        else:
            ht = self.state_encoder(states)
        htH = self.target_state_encoder(next_states)
        htH_hat = self.dynamics(torch.cat((ht, skill), dim = -1))
        
        rewards_pred = self.reward_function(torch.cat((ht, G, skill), dim = -1)).squeeze(-1) 

        state_consistency = F.mse_loss(htH_hat, htH)
        reward_loss = F.mse_loss(rewards_pred, batch.rewards) # real reward 


        policy_skill =  self.highlevel_policy.dist(torch.cat((ht.clone().detach(), relabeled_G), dim = -1))

        GCSL_loss = nll_dist(
            batch.actions,
            policy_skill,
            batch.actions_normal,
            tanh = self.cfg.tanh
        ).mean()

        return  edict(
            state_consistency = state_consistency * 2,
            reward_loss = reward_loss * 0.5,
            GCSL_loss = GCSL_loss,
        )
    
    def rollout_latent(self, inputs):

        ht, skills = inputs['states'], inputs['actions']
        dynamics_input = torch.cat((ht, skills), dim = -1)

        next_states = self.dynamics(dynamics_input)
        rewards_pred = self.reward_function(dynamics_input) 

        return edict(
            next_states = next_states,
            rewards_pred = rewards_pred
        )

    def _std_decay(self, step):
        # from rolf
        mix = np.clip(step / self.cfg.step_interval, 0.0, 1.0)
        return 0.5 * (1-mix) + 0.05 * mix

    def _horizon_decay(self, step):
        # from rolf
        mix = np.clip(step / self.cfg.step_interval, 0.0, 1.0)
        return 1 * (1-mix) + self.cfg.planning_horizon * mix

    def score_weighted_skills(self, loc, score, skills):
        weighted_loc = (score.unsqueeze(-1) * skills).sum(dim=0)
        weighted_std = torch.sqrt(torch.sum(score.unsqueeze(-1) * (skills - weighted_loc.unsqueeze(0)) ** 2, dim=0))
        # soft update 
        new_loc = self.cfg.cem_momentum * loc + (1 - self.cfg.cem_momentum) * weighted_loc
        log_scale = torch.clamp(weighted_std, self._std_decay(self.rollout_step), 2).log() 
        dist = get_dist(new_loc, log_scale, tanh = self.tanh)

        return dist, new_loc
    
    def encode(self, states, keep_grad = False, prior = False):
        if prior:
            if self.cfg.env_name == "kitchen":
                return states[:, :self.cfg.n_pos]
            else:
                return states
            
        else:
            if keep_grad:
                ht = self.state_encoder(states)
            else:
                with torch.no_grad():
                    ht = self.state_encoder(states)
            return ht 

    def __encode__(self, states, keep_grad = False):

        if keep_grad:
            ht = self.state_encoder(states)

        else:
            with torch.no_grad():
                ht = self.state_encoder(states)

        return ht


    def get_rl_params(self):
        rl_params =  edict(
            policy = [ 
                {"params" :  self.highlevel_policy.parameters(), "lr" : self.cfg.policy_lr}
                ],
            consistency = {
                "state" : {
                    "params" : self.state_encoder.parameters(),
                    "lr" : self.cfg.consistency_lr, #self.cfg.invD_lr, 
                    "metric" : None,
                    # "metric" : "skill_consistency"
                    },
                "D" : {
                    "params" : self.dynamics.parameters(), 
                    "lr" : self.cfg.consistency_lr, #self.cfg.D_lr, 
                    "metric" : None,
                    # "metric" : "state_consistency"
                    },
                "R" : {
                    "params" :  self.reward_function.parameters(), 
                    "lr" : 0.001, 
                    # "metric" : "GCSL_loss"
                    "metric" : None,
                },
                "high_policy" : {
                    "params" :  self.highlevel_policy.parameters(), 
                    "lr" : self.cfg.gcsl_lr, 
                    # "metric" : "GCSL_loss"
                    "metric" : None,
                }
            }
                
        )
        
        
        return rl_params

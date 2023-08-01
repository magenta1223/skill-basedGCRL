import numpy as np
import torch
import copy
from easydict import EasyDict as edict
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import update_moving_average
from ...contrib import TanhNormal
from torch.nn import functional as F

class Skimo_Prior(ContextPolicyMixin, BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        super().__init__(submodules)
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.step = 0
        self.qfs = None

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
        states_repr = self.state_encoder(states.view(N * T, -1))
        states_hat = self.state_decoder(states_repr).view(N, T, -1)

            # G = self.state_encoder(G)
        hts = states_repr.view(N, T, -1).clone().detach()
        ht = hts[:, 0]

        with torch.no_grad():
            htH = self.target_state_encoder(states[:, -1])

        # -------------- State-Conditioned Prior -------------- #
        if self.cfg.env_name == "kitchen":
            prior, prior_detach = self.skill_prior.dist(states[:, 0, :self.cfg.n_pos], detached = True)
        else:
            prior, prior_detach = self.skill_prior.dist(states[:, 0], detached = True)

        # ------------------ Skill Dynamics ------------------- #
        dynamics_input = torch.cat((ht, skill), dim = -1)
        D = self.dynamics(dynamics_input)

        # -------------- High-level policy -------------- #
        policy_skill =  self.highlevel_policy.dist(torch.cat((ht.clone().detach(), G), dim = -1))

        # -------------- Rollout for metric -------------- #
        # dense execution with loop (for metric)
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

    
    # @torch.no_grad()
    # def act_cem(self, states, G):
    #     batch = edict(
    #         states = prep_state(states, self.device),
    #         G = prep_state(G, self.device),
    #     )
        
    #     skill_normal, skill = self.cem_planning(batch)
    #     return to_skill_embedding(skill_normal), to_skill_embedding(skill)


    @torch.no_grad()
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 

        batch = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )
        
        if self.qfs is not None:
            skill_normal, skill = self.cem_planning(batch)
            return to_skill_embedding(skill_normal), to_skill_embedding(skill)

            # dist = self.dist(batch, mode = "act").policy_skill
            # if isinstance(dist, TanhNormal):
            #     z_normal, z = dist.sample_with_pre_tanh_value()
            #     return to_skill_embedding(z_normal), to_skill_embedding(z)
            # else:
            #     return None, to_skill_embedding(dist.sample())
        else:
            dist = self.dist(batch, mode = "act").policy_skill
            if isinstance(dist, TanhNormal):
                z_normal, z = dist.sample_with_pre_tanh_value()
                return to_skill_embedding(z_normal), to_skill_embedding(z)
            else:
                return None, to_skill_embedding(dist.sample())

    
    def dist(self, batch, mode = "policy", latent = False):
        """
        latent : for backpropagation through time
        """

        if mode == "consistency":
            return self.consistency(batch)
        
        elif mode == "policy":
            state, G = batch.states, batch.G

            if state.shape[-1] == self.cfg.latent_state_dim:
                ht = state 
            else:
                with torch.no_grad():
                    ht = self.state_encoder(state)
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1))
            return edict(
                policy_skill = policy_skill
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
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            # step을 추가하자
            next_state = self.dynamics( torch.cat((state, skills[:, t]), dim  = -1))
            reward = self.reward_function( torch.cat((state, G, skills[:, t]), dim  = -1)).squeeze(-1) 

            value += discount * reward
            discount *= self.cfg.discount

            state = next_state
        
        policy_skill =  self.highlevel_policy.dist(torch.cat((state, G), dim = -1)).sample()
        q_values = [  qf(torch.cat((state, G, policy_skill), dim = -1)).squeeze(-1)   for qf in qfs]
        
        value += discount * torch.min(*q_values) # 마지막엔 Q에 넣어서 value를 구함. 
        return value

    @torch.no_grad()
    def rollout(self, batch):
        # ht, G = inputs['states'], inputs['G']
        # planning_horizon = inputs['planning_horizon']

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
        """
        Cross Entropy Method
        """

        # planning_horizon  = int(self._horizon_decay(self._step))
        planning_horizon = self.cfg.planning_horizon
        
        qfs = self.qfs 
        states = batch.states
        # state encode
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

        states, next_states, skill, G = batch.states, batch.next_states, batch.actions, batch.G
        
        if states.shape[1] == self.cfg.latent_state_dim:
            ht = states 
        else:
            ht = self.state_encoder(states)
        htH = self.target_state_encoder(next_states)
        htH_hat = self.dynamics(torch.cat((ht, skill), dim = -1))
        
        rewards_pred = self.reward_function(torch.cat((ht, G, skill), dim = -1)).squeeze(-1) 

        state_consistency = F.mse_loss(htH_hat, htH)
        reward_loss = F.mse_loss(rewards_pred, batch.rewards)

        
        return  edict(
            state_consistency = state_consistency * 2,
            reward_loss = reward_loss * 0.5,
            htH_hat = htH_hat,
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
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 0.5 * (1-mix) + 0.01 * mix

    def _horizon_decay(self, step):
        # from rolf
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 1 * (1-mix) + self.planning_horizon * mix


    def score_weighted_skills(self, loc, score, skills):
        weighted_loc = (score.unsqueeze(-1) * skills).sum(dim=0)
        weighted_std = torch.sqrt(torch.sum(score.unsqueeze(-1) * (skills - weighted_loc.unsqueeze(0)) ** 2, dim=0))
        
        # soft update 
        new_loc = self.cfg.cem_momentum * loc + (1 - self.cfg.cem_momentum) * weighted_loc
        # log_scale = torch.clamp(weighted_std, self._std_decay(self._step), 2).log() # new_std의 최소값. .. 을 해야돼? 
        log_scale = weighted_std.log()
        dist = get_dist(new_loc, log_scale, tanh = self.tanh)

        return dist, new_loc
    
    def encode(self, states, keep_grad = False):
        """
        1. latent state가 들어오면 -> 그대로 내보내기 
        2. 아니면 -> 인코딩
        """
        if states.shape[-1] == self.cfg.latent_state_dim:
            return states
        
        else:
            return self.__encode__(states, keep_grad)
        
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
                    "lr" : self.cfg.invD_lr, 
                    "metric" : None,
                    # "metric" : "skill_consistency"
                    },
                "D" : {
                    "params" : self.dynamics.parameters(), 
                    "lr" : self.cfg.D_lr, 
                    "metric" : None,
                    # "metric" : "state_consistency"
                    },
                "R" : {
                    "params" :  self.reward_function.parameters(), 
                    "lr" : 0.001, 
                    # "metric" : "GCSL_loss"
                    "metric" : None,
                }



                # "state_enc" : {
                #     "params" :  self.state_encoder.parameters(), 
                #     "lr" : 1e-6, # now best : 1e-6
                #     # "metric" : "GCSL_loss"
                #     "metric" : None,
                # }

            }
                
        )

        # if self.cfg.with_gcsl:
        #     rl_params["f"] = {
        #         "params" :  self.subgoal_generator.parameters(), 
        #         "lr" : self.cfg.f_lr, 
        #         # "metric" : "GCSL_loss"
        #         "metric" : None,
        #     }

        return rl_params

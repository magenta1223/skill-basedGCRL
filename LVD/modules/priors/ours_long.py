# import copy
from copy import deepcopy
import torch
import torch.distributions as torch_dist
from torch.nn import functional as F
import random
from ...contrib import TanhNormal
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import update_moving_average
from easydict import EasyDict as edict
from .ours_sep import GoalConditioned_Diversity_Sep_Prior

class Ours_LongSkill_Prior(GoalConditioned_Diversity_Sep_Prior):
    """
    """
    def __init__(self, **submodules):
        super().__init__(**submodules)

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # soft update
        if self.n_soft_update % self.update_freq == 0 and self.cfg.phase != "rl":
            # update_moving_average(self.target_state_encoder, self.state_encoder, self.cfg.tau)
            update_moving_average(self.target_state_encoder, self.state_encoder)
            # update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        
        # hard update 
        # update_moving_average(self.target_state_decoder, self.state_decoder, 1)
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        if self.cfg.only_flatD:
            update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)

        # update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics)
        # update_moving_average(self.target_dynamics, self.dynamics)

        self.n_soft_update += 1

    def forward(self, batch, *args, **kwargs):
        states, G = batch.states, batch.G
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr, ht_pos, ht_nonPos = self.state_encoder(states.view(N * T, -1)) # N * T, -1 
            
        states_hat, pos_hat, nonPos_hat = self.state_decoder(states_repr)
        states_hat = states_hat.view(N, T, -1) 
        hts = states_repr.view(N, T, -1).clone()

        with torch.no_grad():
            # target for dynamics & subgoal generator 
            hts_target, _, _ = self.target_state_encoder(states.view(N * T, -1))
            hts_target = hts_target.view(N, T, -1)
            subgoal_target = hts_target[:, 10]
            D_target = hts_target[:, -1]

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.forward_prior(hts)

        # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts[:, -1])
        
        if self.cfg.grad_swap:
            invD_skill_normal, invD_skill = inverse_dynamics.rsample_with_pre_tanh_value()
            skill = invD_skill - invD_skill.clone().detach() + batch.skill

        elif self.cfg.grad_pass.skill:
            invD_skill_normal, invD_skill = inverse_dynamics.rsample_with_pre_tanh_value()
            skill = batch.skill
        else:
            invD_skill_normal, invD_skill = inverse_dynamics.rsample_with_pre_tanh_value()
            skill = invD_skill.clone()

        # -------------- Dynamics Learning -------------- #                
        flat_D, cache = self.forward_flatD(hts, batch.skill)
        D = self.forward_D(hts[:, 0], skill)
        
        if self.cfg.manipulation:
            diff_nonPos_latent = cache[-1].view(N, T-1, -1)
            diff = self.diff_decoder(diff_nonPos_latent.clone().detach())
            diff_target = states[:, 1:, self.cfg.n_pos:] - states[:, :-1, self.cfg.n_pos:]
        else:
            diff = None
            diff_target = None



        # -------------- Subgoal Generator -------------- #
        invD_sub, subgoal_D, subgoal_f = self.forward_subgoal_G(hts[:, 0], G)
        
        # -------------- High Policy -------------- #
        if self.cfg.learning_mode == "only_skill":
            policy_skill = self.high_policy.dist(torch.cat((hts[:, 0].clone().detach(), G), dim = -1))
        else:
            policy_skill = None



        result = edict(
            # State Auto-Encoder
            states = states,
            states_hat = states_hat,
            states_repr = states_repr,

            # Skills 
            prior = prior,
            prior_detach = prior_detach,
            invD = inverse_dynamics,
            invD_detach = inverse_dynamics_detach,
            # invD_target = invD_target,

            # Dynamics modules
            D = D,
            flat_D = flat_D,
            D_target =  D_target, 
            flat_D_target = hts_target[:, 1:],

            # Subgoal generator 
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f,
            subgoal_D_target =  subgoal_f,
            subgoal_f_target =  subgoal_target,
            
            # high_policy
            policy_skill = policy_skill,

            # Difference Decoder for manipulation task 
            diff = diff,
            diff_target = diff_target,
            invD_sub = invD_sub,
        )
        
        # -------------- Rollout for metric -------------- #
        if not self.training:
            check_subgoals_input = (hts, skill, D, subgoal_f)
            subgoals = self.check_subgoals(check_subgoals_input)
            result.update(subgoals)

        return result
    

    def forward_subgoal_G(self, start, G):

        start_detached = start.clone().detach() # stop grad : 안하면 goal과의 연관성이 너무 심해짐. 
        start_original = start.clone().detach()
        
        for _ in range((self.cfg.subseq_len - 1) // 10):
            sg_input = self.sg_input(start_detached, G)

            _subgoal_f = self.subgoal_generator(sg_input)
            if self.cfg.sg_residual:
                subgoal_f = _subgoal_f + start_detached
            else:
                subgoal_f = _subgoal_f
            start_detached = subgoal_f
                
            
            
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start_original, subgoal= subgoal_f, tanh = self.cfg.tanh)

        _, skill_sub = invD_sub.rsample_with_pre_tanh_value()
            
        if self.cfg.only_flatD:
            for _ in range(self.cfg.subseq_len - 1):
                start_detached, _ = self.forward_flatD(start_original, skill_sub, use_target= True)
            subgoal_D = start_detached 
        else:
            subgoal_D = self.forward_D(start_detached, skill_sub, use_target= True)

        return invD_sub, subgoal_D, subgoal_f
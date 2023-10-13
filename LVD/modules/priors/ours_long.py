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
from .ours import Ours_Prior

class Ours_LongSkill_Prior(Ours_Prior):
    """
    """
    def __init__(self, **submodules):
        super().__init__(**submodules)


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
            D_target = hts_target[:, -1]
            
            # skill and subgoal have different lengths
            subgoal_target = hts_target[:, self.cfg.subseq_len]

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

        # -------------- Skill-step goal Generator -------------- #
        invD_sub, subgoal_D, subgoal_f_10, subgoal_f_long = self.forward_skill_step_G(hts[:, 0], G)
        
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

            # Dynamics modules
            D = D,
            flat_D = flat_D,
            D_target =  D_target, 
            flat_D_target = hts_target[:, 1:],

            # Subgoal generator 
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f_10,
            subgoal_D_target =  subgoal_f_long,
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
            check_subgoals_input = (hts, skill, D, subgoal_f_long)
            subgoals = self.check_subgoals(check_subgoals_input)
            result.update(subgoals)

        return result
    

    def forward_skill_step_G(self, start, G):

        start_detached = start.clone().detach() 
        start_original = start.clone().detach()
        

        subgoals_f = []
        for _ in range((21 - 1) // 10):
            # infer subgoal
            sg_input = self.sg_input(start_detached, G)
            _subgoal_f = self.skill_step_goal_generator(sg_input)
            if self.cfg.sg_residual:
                subgoal_f = _subgoal_f + start_detached
            else:
                subgoal_f = _subgoal_f
            # next subgoal
            start_detached = subgoal_f
            subgoals_f.append(subgoal_f)
                    
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start_original, subgoal= subgoal_f, tanh = self.cfg.tanh)

        _, skill_sub = invD_sub.rsample_with_pre_tanh_value()
            
        if self.cfg.only_flatD:
            for _ in range(self.cfg.subseq_len - 1):
                start_detached, _ = self.forward_flatD(start_original, skill_sub, use_target= True)
            subgoal_D = start_detached 
        else:
            subgoal_D = self.forward_D(start_detached, skill_sub, use_target= True)


        # invD_sub : skill for \tau_0:20
        # subgoal_D : subgoal from dynamics with 20 (subseq_len = 21)
        # subgoals_f[0] : subgoal from f with time step 10 for BC term in Eq. subgoal
        # subgoals_f[1] : subgoal from f with time step 20 for sanity check
        return invD_sub, subgoal_D,  subgoals_f[0], subgoals_f[1]  #z_0:20, h_20, h_10 
    
    def dist(self, batch, mode = "policy"):
        assert mode in ['policy', 'consistency', 'act', 'rollout'], "Invalid mode"
        if mode == "consistency":
            return self.consistency(batch)
 
        elif mode == "act":
            # act
            state, G = batch.states, batch.G 
            with torch.no_grad():
                ht, ht_pos, ht_nonPos = self.state_encoder(state)

            if self.cfg.learning_mode == "only_skill":
                policy_skill = self.high_policy.dist(torch.cat((ht, G), dim = -1))
                result =  edict(
                    policy_skill = policy_skill
                )    

            else:
                invD, subgoal_D, subgoal_f, subgoal_f_long = self.forward_skill_step_G(ht, G)
                result =  edict(
                    policy_skill = invD
                )    

            return result 
        
        elif mode == "rollout":
            # latent state가 
            with torch.no_grad():
                _, subgoal_D, _ = self.forward_skill_step_G(batch.ht, batch.G)
            
            return edict(
                ht = subgoal_D 
            )

        else:
            # policy 
            states, G = batch.states, batch.G
            with torch.no_grad():
                ht, ht_pos, ht_nonPos = self.state_encoder(states)

            # forward subgoal generator 
            sg_input = self.sg_input(ht, G)
            
            if self.cfg.sg_residual:
                subgoal_f = self.skill_step_goal_generator(sg_input)
                subgoal_f = subgoal_f + ht
            
            # skill inference 
            invD, _ = self.forward_invD(ht, subgoal_f)
            skill = invD.rsample() 
            
            # skill execution
            D = self.forward_D(ht, skill)
            
            state_consistency_f = F.mse_loss(subgoal_f.detach(), D)
            # assert 1==0, state_consistency_f.item()
            result =  edict(
                policy_skill = invD,
                additional_losses = dict(
                    state_consistency_f = state_consistency_f
                )
            )    

            return result
    
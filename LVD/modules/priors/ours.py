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


class Ours_Prior(ContextPolicyMixin, BaseModule):
    """
    """
    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_state_encoder = deepcopy(self.state_encoder)
        # self.target_state_decoder = deepcopy(self.state_decoder)
        self.target_inverse_dynamics = deepcopy(self.inverse_dynamics)
        self.target_dynamics = deepcopy(self.dynamics)
            
        if self.cfg.only_flatD:
            self.target_flat_dynamics = deepcopy(self.flat_dynamics)


        self.n_soft_update = 1
        self.update_freq = 5
        self.state_processor = StateProcessor(self.cfg.env_name)

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # soft update
        if self.n_soft_update % self.update_freq == 0 and self.cfg.phase != "rl":
            update_moving_average(self.target_state_encoder, self.state_encoder)
        
        # hard update 
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        if self.cfg.only_flatD:
            update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)

        self.n_soft_update += 1

    def forward(self, batch, *args, **kwargs):
        states, G = batch.states, batch.G
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr, _, _ = self.state_encoder(states.view(N * T, -1)) # N * T, -1 
        states_hat, _, _ = self.state_decoder(states_repr)
        states_hat = states_hat.view(N, T, -1) 
        hts = states_repr.view(N, T, -1).clone()

        with torch.no_grad():
            # target for dynamics & subgoal generator 
            hts_target, _, _ = self.target_state_encoder(states.view(N * T, -1))
            hts_target = hts_target.view(N, T, -1)
            subgoal_target = hts_target[:, -1]

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.forward_prior(hts)

        # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts[:, -1])
        
        invD_skill_normal, invD_skill = inverse_dynamics.rsample_with_pre_tanh_value()
        skill = invD_skill - invD_skill.clone().detach() + batch.skill

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
        invD_sub, subgoal_D, subgoal_f = self.forward_skill_step_G(hts[:, 0], G)
        
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
            D_target =  subgoal_target, 
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
        check_subgoals_input = (hts, skill, D, subgoal_f)
        subgoals = self.check_subgoals(check_subgoals_input)
        result.update(subgoals)
        return result
    
    def forward_prior(self, start):
        if len(start.shape) > 2:
            start = start[:, 0]
        
        if self.cfg.manipulation:
            start = start.chunk(2, -1)[0].clone().detach()
            return self.skill_prior.dist(start, detached = True)
        else:
            # 
            return self.skill_prior.dist(start.clone().detach(), detached = True)

    def forward_invD(self, start, subgoal):
        start = start.clone().detach()
        subgoal = subgoal.clone().detach() 

        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.cfg.tanh)
        
        return inverse_dynamics, inverse_dynamics_detach
    
    def forward_flatD(self, start, skill, use_target = False):
        if not self.cfg.grad_pass.flat_D:
            start = start.clone().detach()

        # rollout / check 
        pos, nonPos = start.chunk(2, -1)

        # rollout 
        if len(pos.shape) < 3:
            flat_dynamics_input = torch.cat((start, skill), dim=-1)
            
            if use_target:
                flat_D = self.target_flat_dynamics(flat_dynamics_input)                
            else:
                flat_D = self.flat_dynamics(flat_dynamics_input)
                
            pos_now, nonPos_now = flat_D.chunk(2, -1)

            if self.cfg.diff:
                flat_D = start + flat_D

        # learning. 
        else:
            N, T = pos.shape[:2]
            skill_length = T- 1
            # skill 
            skill = skill.unsqueeze(1).repeat(1, skill_length, 1)

            flat_dynamics_input = torch.cat((start[:, :-1], skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)
            
            if self.cfg.diff.flat:
                if self.cfg.manipulation:
                    pos_now, nonPos_now = flat_D.chunk(2, -1)
                else:
                    # pos_now, nonPos_now = flat_D, None
                    pos_now, nonPos_now = flat_D.chunk(2, -1)

            else:
                pos_now, nonPos_now = None, flat_D.chunk(2, -1)[1] - nonPos[:, :-1]

            if self.cfg.diff.flat:
                flat_D = start[:,:-1] + flat_D


        cache = pos, nonPos, pos_now, nonPos_now

        return flat_D, cache

    def forward_D(self, start, skill, use_target = False):

        dynamics_input = torch.cat((start, skill), dim=-1)

        if use_target:
            D = self.target_dynamics(dynamics_input)
        else:
            D = self.dynamics(dynamics_input)

        if self.cfg.diff.skill:
            if self.cfg.grad_pass.skill_D:
                D = start + D
            else:
                D = start.clone().detach() + D

        return D
    
    def sg_input(self, start, G):
        return torch.cat((start, G), dim = -1)

    def forward_skill_step_G(self, start, G):
        start_detached = start.clone().detach() 
        sg_input = self.sg_input(start_detached, G)
        _subgoal_f = self.skill_step_goal_generator(sg_input)
        subgoal_f = _subgoal_f + start_detached

        invD_sub, _ = self.target_inverse_dynamics.dist(state = start_detached, subgoal= subgoal_f, tanh = self.cfg.tanh)

        _, skill_sub = invD_sub.rsample_with_pre_tanh_value()
            
        if self.cfg.only_flatD:
            for _ in range(self.cfg.subseq_len - 1):
                start_detached, _ = self.forward_flatD(start_detached, skill_sub, use_target= True)
            subgoal_D = start_detached 
        else:
            subgoal_D = self.forward_D(start_detached, skill_sub, use_target= True)

        return invD_sub, subgoal_D, subgoal_f
    
    # for evalutation 
    @torch.no_grad()
    def check_subgoals(self, inputs):
        hts, skill, D, subgoal_f = inputs
        skill_length = hts.shape[1] - 1
        start = hts[:,0]
        _ht = start.clone()
            
        for _ in range(skill_length):
            _ht, _ = self.forward_flatD(_ht, skill)


        subgoal_recon_D, _, _ = self.state_decoder(D)
        subgoal_recon_f, _, _ = self.state_decoder(subgoal_f)

        return edict(
            subgoal_rollout =  _ht,
            subgoal_recon_D = subgoal_recon_D,
            subgoal_recon_f =  subgoal_recon_f,
        )
    
    
    @torch.no_grad()
    def rollout(self, batch):
        """
        """
        # requirements 
        states = batch.states
        N, T, _ = states.shape
        skill_length = T - 1
        plan_H = batch['plan_H']

        states_rollout = []
        latent_states_rollout = []
        skills = [] 

        # select branching point 
        c = random.sample(range(1, skill_length - 1), 1)[0]
        _state = states[:, c]
        _ht, _, _ = self.state_encoder(_state)
        # for robotics 
        nonPos_raw_state = states[:, c, self.cfg.n_pos:]

        # sample skill 
        skill = self.forward_prior(_ht)[0].sample()
        
        
        # rollout start 
        for i in range(plan_H):
            # execute skill on latent space and append to the original sub-trajectory 
            if (i - c) % self.cfg.sample_interval == 0: # skill sample frequenct m = 5
                skill = self.forward_prior(_ht)[0].sample()
            
            # skill execute 
            next_ht, cache = self.forward_flatD(_ht, skill) 
        
            # decode latent state into raw state
            if self.cfg.manipulation:
                _, _, _, diff_nonPos_latent = cache
                diff_nonPos = self.diff_decoder(diff_nonPos_latent)
                _, pos_raw_state, _ = self.state_decoder(next_ht)
                nonPos_raw_state = nonPos_raw_state + diff_nonPos
                _state = torch.cat((pos_raw_state, nonPos_raw_state), dim = -1)

            else:
                _state, pos_raw_state, _ = self.state_decoder(next_ht)
                
            # append 
            states_rollout.append(_state)
            latent_states_rollout.append(_ht)
            skills.append(skill)
            
            # next state
            _ht = next_ht
            
        states_rollout = torch.stack(states_rollout, dim = 1)
        latent_states_rollout = torch.stack(latent_states_rollout, dim = 1)
        skills = torch.stack(skills, dim = 1)
        
        
        result =  edict(
            c = c,
            states_rollout = states_rollout,
            latent_states_rollout = latent_states_rollout,
            skills = skills,
        )


        return result 

    def encode(self, states, keep_grad = False, prior = False):
        with torch.no_grad():
            if prior and self.cfg.manipulation:
                ht, ht_pos, ht_nonPos = self.state_encoder(states)
                ht = ht_pos
            else:
                ht, _, _ = self.state_encoder(states)

        return ht

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
                invD, subgoal_D, subgoal_f = self.forward_skill_step_G(ht, G)
                result =  edict(
                    policy_skill = invD,
                    subgoal_f = subgoal_f
                )    

            return result 
        
        elif mode == "rollout":
            with torch.no_grad():
                _, subgoal_D, _ = self.forward_skill_step_G(batch.ht, batch.G)
            
            return edict(
                ht = subgoal_D 
            )

        else:
            return self.policy_dist(batch)
        
    def policy_dist(self, batch):
        # policy 
        states, G = batch.states, batch.G
        with torch.no_grad():
            ht, ht_pos, ht_nonPos = self.state_encoder(states)

        if self.cfg.learning_mode == "only_skill":
            policy_skill = self.high_policy.dist(torch.cat((ht, G), dim = -1))
            result =  edict(
                policy_skill = policy_skill,
                additional_losses = dict(
                )
            )    

        else:
            start_detached = ht.clone().detach()             
            invD, D, subgoal_f = self.forward_skill_step_G(ht, G)
            if self.cfg.cyclic_consistency:
                state_consistency_f = F.mse_loss(subgoal_f, D)
            else:
                state_consistency_f = F.mse_loss(subgoal_f, D).detach()
                        
            result =  edict(
                policy_skill = invD,
                additional_losses = dict(
                    state_consistency_f = state_consistency_f
                )
            )    

        return result
    
    @torch.no_grad()
    def act(self, states, G):
        batch = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(batch, mode = "act").policy_skill

        if isinstance(dist, TanhNormal):
            if self.explore: # collect ep for adaptation 
                z_normal, z = dist.sample_with_pre_tanh_value()
                return to_skill_embedding(z_normal), to_skill_embedding(z)
            else: # evaluation 
                return to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2)), to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2))

        else:
            return None, to_skill_embedding(dist.sample())
        
    def consistency(self, batch):
        """
        Finetune inverse dynamics and dynamics with the data collected in online.
        """
        states, G, next_states = batch.states, batch.relabeled_G, batch.next_states

        ht, _, _ = self.state_encoder(states)
        htH,  _, _ = self.state_encoder(next_states)
        htH_target,  _, _ = self.target_state_encoder(next_states)
        
        if self.cfg.learning_mode == "only_skill":
            policy_skill = self.high_policy.dist(torch.cat((ht, G), dim = -1))

            skill_consistency = nll_dist(
                batch.actions,
                policy_skill,
                batch.actions_normal,
                tanh = self.cfg.tanh
            ).mean()
            
            return  edict(
                skill_consistency = skill_consistency
            )

        else:
            invD, invD_detach = self.forward_invD(ht, htH)
            D = self.forward_D(ht, batch.actions)
            
            state_consistency = F.mse_loss(D, htH_target) #+ mmd_loss            
            skill_consistency = nll_dist(
                batch.actions,
                invD,
                batch.actions_normal,
                # tanh = self.tanh
                tanh = self.cfg.tanh
            ).mean()

            invD_sub, subgoal_D, subgoal_f = self.forward_skill_step_G(ht, G)
                        
            # used for metric
            GCSL_loss = F.mse_loss(subgoal_D, htH_target) + F.mse_loss(subgoal_f, subgoal_D)
            GCSL_loss = GCSL_loss.detach()

        
            if not self.cfg.consistency_update:
                state_consistency = state_consistency.detach()
                skill_consistency = skill_consistency.detach()
        
            return  edict(
                state_consistency = state_consistency,
                skill_consistency = skill_consistency,
                GCSL_loss = GCSL_loss
            )


    def get_rl_params(self):
        if self.cfg.learning_mode == "only_skill":
            rl_params =  edict(
                policy = [ 
                    {"params" :  self.high_policy.parameters(), "lr" : self.cfg.policy_lr}
                    ],
                consistency = {
                    "high_policy" : {
                        "params" : self.high_policy.parameters(),
                        "lr" : self.cfg.consistency_lr, 
                        "metric" : None,
                        },
                }
            )
            
            return rl_params            
            
        else:
            rl_params =  edict(
                policy = [ 
                    {"params" :  self.skill_step_goal_generator.parameters(), "lr" : self.cfg.policy_lr}
                    ],
                consistency = {
                    "invD" : {
                        "params" : self.inverse_dynamics.parameters(),
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
                }
                    
            )
                            
            return rl_params

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


class GoalConditioned_Diversity_Joint_Sep_Prior(ContextPolicyMixin, BaseModule):
    """
    """
    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_state_encoder = deepcopy(self.state_encoder)
        # self.target_state_decoder = deepcopy(self.state_decoder)
        self.target_inverse_dynamics = deepcopy(self.inverse_dynamics)
        self.target_dynamics = deepcopy(self.dynamics)
        # self.target_flat_dynamics = deepcopy(self.flat_dynamics)


        self.n_soft_update = 1
        self.update_freq = 5
        self.state_processor = StateProcessor(self.cfg.env_name)

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # soft update
        if self.n_soft_update % self.update_freq == 0:
            # update_moving_average(self.target_state_encoder, self.state_encoder, self.cfg.tau)
            update_moving_average(self.target_state_encoder, self.state_encoder)
        
        # hard update 
        # update_moving_average(self.target_state_decoder, self.state_decoder, 1)
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        # update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)

        self.n_soft_update += 1

    def forward(self, batch, *args, **kwargs):
        states, G = batch.states, batch.G
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr, ht_pos, ht_nonPos = self.state_encoder(states.view(N * T, -1)) # N * T, -1 
        scales = 0
            
        states_hat, pos_hat, nonPos_hat = self.state_decoder(states_repr)
        states_hat = states_hat.view(N, T, -1) 
        hts = states_repr.view(N, T, -1).clone()

        with torch.no_grad():
            # target for dynamics & subgoal generator 
            hts_target, _, _ = self.target_state_encoder(states.view(N * T, -1))
            hts_target = hts_target.view(N, T, -1)
            subgoal_target = hts_target[:, -1]

        # -------------- State-Conditioned Prior -------------- #
        if self.cfg.manipulation:
            prior, prior_detach = self.skill_prior.dist(ht_pos.view(N, T, -1)[:, 0].clone().detach(), detached = True)
        else:
            prior, prior_detach = self.skill_prior.dist(hts[:, 0].clone().detach(), detached = True)

        # # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts[:,-1])
        skill = inverse_dynamics.rsample()

        # # -------------- Dynamics Learning -------------- #                
        flat_D, cache = self.forward_flatD(hts, skill)
        D = self.forward_D(hts[:, 0], skill)
        
        # only env diff만 
        if self.cfg.manipulation:
            diff_nonPos_latent = cache[-1].view(N, T-1, -1)
            diff = self.diff_decoder(diff_nonPos_latent.clone().detach())
            diff_target = states[:, 1:, self.cfg.n_pos:] - states[:, :-1, self.cfg.n_pos:]
        else:
            # diff = 0
            # diff_target = 0
            diff_nonPos_latent = cache[-1].view(N, T-1, -1)
            diff = self.diff_decoder(diff_nonPos_latent.clone().detach())
            diff_target = states[:, 1:, self.cfg.n_pos:] - states[:, :-1, self.cfg.n_pos:]


        # # -------------- Subgoal Generator -------------- #
        invD_sub, subgoal_D, subgoal_f = self.forward_subgoal_G(hts[:, 0], G)

        # -------------- Rollout for metric -------------- #
        check_subgoals_input = (hts, inverse_dynamics, D, subgoal_D, subgoal_f)
        subgoals = self.check_subgoals(check_subgoals_input)

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
            D_target =  subgoal_target, 
            flat_D_target = hts_target[:, 1:],

            # Subgoal generator 
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f,
            subgoal_target =  subgoal_target,

            # Difference Decoder for manipulation task 
            diff = diff,
            diff_target = diff_target,
            invD_sub = invD_sub,
            scales = scales
        )
        
        # metrics 
        result.update(subgoals)
        return result
    
    # def forward_invD(self, hts):
    def forward_invD(self, start, subgoal):
        if not self.cfg.grad_pass.invD:
            start = start.clone().detach()
            subgoal = subgoal.clone().detach()

        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.cfg.tanh)

        return inverse_dynamics, inverse_dynamics_detach
    
    def forward_flatD(self, start, skill):

        # rollout / check 
        if self.cfg.manipulation:
            pos, nonPos = start.chunk(2, -1)
        else:
            pos, nonPos = start, None
        # pos, nonPos = start.chunk(2, -1)

        if len(pos.shape) < 3:
            flat_dynamics_input = torch.cat((pos, skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input)
            pos_now, nonPos_now = flat_D.chunk(2, -1)

            if self.cfg.grad_pass.D:
                flat_D = start + flat_D
            else:
                flat_D = start.clone().detach() + flat_D

        else:
            N, T = pos.shape[:2]
            skill_length = T- 1
            skill = skill.unsqueeze(1).repeat(1, skill_length, 1)
            flat_dynamics_input = torch.cat((pos[:, :-1], skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)
            
            if self.cfg.manipulation:
                pos_now, nonPos_now = flat_D.chunk(2, -1)
            else:
                # pos_now, nonPos_now = flat_D, None
                pos_now, nonPos_now = flat_D.chunk(2, -1)

            if self.cfg.grad_pass.D:
                flat_D = start[:,:-1] + flat_D
            else:
                flat_D = start[:,:-1].clone().detach() + flat_D

        cache = pos, nonPos, pos_now, nonPos_now

        return flat_D, cache

    def forward_D(self, start, skill, use_target = False):

        if self.cfg.manipulation:
            pos, nonPos = start.chunk(2, -1)
        else:
            pos, nonPos = start, None
        # pos, nonPos = start.chunk(2, -1)


        dynamics_input = torch.cat((pos, skill), dim=-1)

        if use_target:
            D = self.target_dynamics(dynamics_input)
        else:
            D = self.dynamics(dynamics_input)

        if self.cfg.grad_pass.D:
            D = start + D
        else:
            D = start.clone().detach() + D

        return D

    def forward_subgoal_G(self, start, G):
        start = start.clone().detach() # stop grad : 안하면 goal과의 연관성이 너무 심해짐. 
        sg_input = torch.cat((start,  G), dim = -1)
        subgoal_f = self.subgoal_generator(sg_input)
        subgoal_f = subgoal_f + start
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start, subgoal = subgoal_f, tanh = self.cfg.tanh)
        skill_sub = invD_sub.rsample()
        subgoal_D = self.forward_D(start, skill_sub, use_target= True)
        return invD_sub, subgoal_D, subgoal_f


    @torch.no_grad()
    def check_subgoals(self, inputs):
        hts, inverse_dynamics, D, subgoal_D, subgoal_f = inputs
        skill_length = hts.shape[1] - 1
        start, subgoal = hts[:,0], hts[:-1]
        _ht = start.clone()
        skill = inverse_dynamics.sample()

        for _ in range(skill_length):
            _ht, _ = self.forward_flatD(_ht, skill)

        invD_rollout_main, _ = self.target_inverse_dynamics.dist(state = start, subgoal = _ht, tanh = self.cfg.tanh)


        subgoal_recon, _, _ = self.state_decoder(subgoal)
        subgoal_recon_D, _, _ = self.state_decoder(D)
        subgoal_recon_D_f, _, _ = self.state_decoder(subgoal_D)
        subgoal_recon_f, _, _ = self.state_decoder(subgoal_f)

        return edict(
            invD_rollout_main= invD_rollout_main,
            subgoal_rollout =  _ht,
            subgoal_recon_D = subgoal_recon_D,
            subgoal_recon_f =  subgoal_recon_f,
            subgoal_recon_D_f =  subgoal_recon_D_f,
            subgoal_recon =  subgoal_recon
        )
    
    @torch.no_grad()
    def rollout(self, batch):
        states = batch.states
        N, T, _ = states.shape
        skill_length = T - 1
        hts, _, _ = self.state_encoder(states.view(N * T, -1))
        hts = hts.view(N, T, -1)  

        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()
        ht_pos, ht_nonPos = _ht.chunk(2, -1)
        # skill_sampled = self.skill_prior.dist(ht_pos).sample()
        
        if self.cfg.manipulation:
            skill_sampled = self.skill_prior.dist(ht_pos).sample()
        else:
            skill_sampled = self.skill_prior.dist(_ht).sample()

        if "skill" in batch:
            skill_sampled = batch.skill 

        states_rollout = []
        _state = states[:, c]
        nonPos_raw_state = states[:, c, self.cfg.n_pos:]

        # 1 skill
        for _ in range(c, skill_length):
            # execute skill on latent space and append to the original sub-trajectory 
            # flat_D로 rollout -> diff in latent space -> diff in raw state space 
            next_ht, cache = self.forward_flatD(_ht, skill_sampled) 

            if self.cfg.manipulation:
                _, _, _, diff_nonPos_latent = cache
                diff_nonPos = self.diff_decoder(diff_nonPos_latent)
                _, pos_raw_state, _ = self.state_decoder(next_ht)
                nonPos_raw_state = nonPos_raw_state + diff_nonPos
                _state = torch.cat((pos_raw_state, nonPos_raw_state), dim = -1)
            else:
                # 여기서는 next_ht가 unseen일까? 아뇨?  
                _state, pos_raw_state, _ = self.state_decoder(next_ht)
                # _state = pos_raw_state


            # _, _, _, diff_nonPos_latent = cache
            # diff_nonPos = self.diff_decoder(diff_nonPos_latent)
            # _, pos_raw_state, _ = self.state_decoder(next_ht)
            # nonPos_raw_state = nonPos_raw_state + diff_nonPos
            # _state = torch.cat((pos_raw_state, nonPos_raw_state), dim = -1)



            states_rollout.append(_state)
            _ht = next_ht
            ht_pos, ht_nonPos = _ht.chunk(2, -1)
        
        skills = []
        # for f learning, execute 4 skill more
        for i in range(self.cfg.plan_H):
            if i % 10 == 0:
                # skill = self.skill_prior.dist(ht_pos).sample()
                if self.cfg.manipulation:
                    skill = self.skill_prior.dist(ht_pos).sample()
                else:
                    skill = self.skill_prior.dist(_ht).sample()

            next_ht, cache = self.forward_flatD(_ht, skill) 
            if self.cfg.manipulation:
                _, _, _, diff_nonPos_latent = cache
                diff_nonPos = self.diff_decoder(diff_nonPos_latent)
                _, pos_raw_state, _ = self.state_decoder(next_ht)
                nonPos_raw_state = nonPos_raw_state + diff_nonPos

                _state = torch.cat((pos_raw_state, nonPos_raw_state), dim = -1)
            else:
                # _, pos_raw_state, _ = self.state_decoder(next_ht)
                # _state = pos_raw_state
                _state, pos_raw_state, _ = self.state_decoder(next_ht)
            # _, _, _, diff_nonPos_latent = cache
            # diff_nonPos = self.diff_decoder(diff_nonPos_latent)
            # _, pos_raw_state, _ = self.state_decoder(next_ht)
            # nonPos_raw_state = nonPos_raw_state + diff_nonPos

            # _state = torch.cat((pos_raw_state, nonPos_raw_state), dim = -1)
    


            # back to ppc 
            states_rollout.append(_state)
            _ht = next_ht
            ht_pos, ht_nonPos = _ht.chunk(2, -1)

            skills.append(skill)
            
        states_rollout = torch.stack(states_rollout, dim = 1)
        skills = torch.stack(skills, dim = 1)

        result =  edict(
            c = c,
            states_rollout = states_rollout,
            skill_sampled = skill_sampled,
            skills = skills,
        )


        return result 

    def encode(self, states, keep_grad = False, prior = False):
        if keep_grad:
            if prior:
                ht, ht_pos, ht_nonPos = self.state_encoder(states)
                ht = ht_pos
            else:
                ht, _, _ = self.state_encoder(states)

        else:
            with torch.no_grad():
                if prior:
                    ht, ht_pos, ht_nonPos = self.state_encoder(states)
                    ht = ht_pos
                else:
                    ht, _, _ = self.state_encoder(states)

        return ht

    def dist(self, batch, mode = "policy"):
        assert mode in ['policy', 'consistency', 'act'], "Invalid mode"
        if mode == "consistency":
            return self.consistency(batch)
        else:
            # policy or act
            state, G = batch.states, batch.G 
            with torch.no_grad():
                ht, ht_pos, ht_nonPos = self.state_encoder(state)

            # subgoal 
            _, _, subgoal_f = self.forward_subgoal_G(ht, G)

            # inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)
            inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.cfg.tanh)
            # inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)
            

            skill = inverse_dynamics_hat.rsample() 
            D = self.forward_D(ht, skill)

            result =  edict(
                policy_skill = inverse_dynamics_hat,
                additional_losses = dict(
                    # state_consistency_f = F.mse_loss(diff_subgoal_f, diff)
                    state_consistency_f = F.mse_loss(subgoal_f, D)

                )
            )    

            return result
    
    @torch.no_grad()
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 

        batch = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(batch, mode = "act").policy_skill

        if isinstance(dist, TanhNormal):
            z_normal, z = dist.sample_with_pre_tanh_value()
            return to_skill_embedding(z_normal), to_skill_embedding(z)
        else:
            return None, to_skill_embedding(dist.sample())
        
    def consistency(self, batch):
        """
        Finetune inverse dynamics and dynamics with the data collected in online.
        """
        # BC를 하려면 반드시 relabeled G여야 함. 
        # states, G, next_states = batch.states, batch.G, batch.next_states
        states, G, next_states = batch.states, batch.relabeled_G, batch.next_states

        # self.state_encoder.eval()

        ht, _, _ = self.state_encoder(states)
        htH,  _, _ = self.state_encoder(next_states)
        htH_target,  _, _ = self.target_state_encoder(next_states)


        # inverse dynamics 
        # if self.grad_pass_invD:
        # if self.cfg.grad_pass.invD:
        #     invD, invD_detach = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.cfg.tanh)
        # else:
        #      invD, invD_detach = self.inverse_dynamics.dist(state = ht.clone().detach(), subgoal = htH.clone().detach(), tanh = self.cfg.tanh)

        invD, invD_detach = self.forward_invD(ht, htH)

        D = self.forward_D(ht, batch.actions)

        # state_consistency = F.mse_loss(ht.clone().detach() + diff, htH) + mmd_loss
        state_consistency = F.mse_loss(D, htH_target) #+ mmd_loss
        skill_consistency = nll_dist(
            batch.actions,
            invD,
            batch.actions_normal,
            # tanh = self.tanh
            tanh = self.cfg.tanh
        ).mean()
        
        # GCSL
        # 똑같은 skill 뽑으면 됨./ 

        invD_sub, subgoal_D, subgoal_f = self.forward_subgoal_G(ht, G)

        GCSL_loss = F.mse_loss(subgoal_f, htH_target) + F.mse_loss(subgoal_D, htH_target) + nll_dist(
            batch.actions,
            invD_sub,
            batch.actions_normal,
            tanh = self.cfg.tanh
        ).mean()
        
        # GCSL_loss = F.mse_loss(diff_subgoal_f + ht, htH) + torch_dist.kl_divergence(invD_sub, invD_detach).mean()
        # GCSL_loss = F.mse_loss(subgoal_f, htH_target) + torch_dist.kl_divergence(invD_sub, invD_detach).mean()



        return  edict(
            state_consistency = state_consistency,
            skill_consistency = skill_consistency,
            GCSL_loss = GCSL_loss if self.cfg.with_gcsl else GCSL_loss.item()
            # GCSL_loss = GCSL_loss
        )


    def get_rl_params(self):
        rl_params =  edict(
            policy = [ 
                {"params" :  self.subgoal_generator.parameters(), "lr" : self.cfg.policy_lr}
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
                # "state_enc" : {
                #     "params" :  self.state_encoder.parameters(), 
                #     "lr" : 1e-6, # now best : 1e-6
                #     # "metric" : "GCSL_loss"
                #     "metric" : None,
                # }

            }
                
        )

        if self.cfg.with_gcsl:
            rl_params["f"] = {
                "params" :  self.subgoal_generator.parameters(), 
                "lr" : self.cfg.f_lr, 
                # "metric" : "GCSL_loss"
                "metric" : None,
            }

        return rl_params

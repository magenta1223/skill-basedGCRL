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


class GoalConditioned_Diversity_Prior(ContextPolicyMixin, BaseModule):
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
        if self.n_soft_update % self.update_freq == 0 and self.cfg.phase != "rl":
            # update_moving_average(self.target_state_encoder, self.state_encoder, self.cfg.tau)
            update_moving_average(self.target_state_encoder, self.state_encoder)
        
        # hard update 
        # update_moving_average(self.target_state_decoder, self.state_decoder, 1)
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        # update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)

        # update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics)
        # update_moving_average(self.target_dynamics, self.dynamics)

        self.n_soft_update += 1

    def forward(self, batch, *args, **kwargs):
        states, G = batch.states, batch.G
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr = self.state_encoder(states.view(N * T, -1)) # N * T, -1 
        scales = 0
            
        states_hat = self.state_decoder(states_repr)
        states_hat = states_hat.view(N, T, -1) 
        hts = states_repr.view(N, T, -1).clone()

        with torch.no_grad():
            # target for dynamics & subgoal generator 
            hts_target = self.target_state_encoder(states.view(N * T, -1))
            hts_target = hts_target.view(N, T, -1)
            subgoal_target = hts_target[:, -1]

        # -------------- State-Conditioned Prior -------------- #
        if self.cfg.manipulation:
            prior, prior_detach = self.skill_prior.dist(hts[:, 0].clone().detach(), detached = True)
            # prior, prior_detach = self.skill_prior.dist(ht_pos.view(N, T, -1)[:, 0], detached = True)

        else:
            prior, prior_detach = self.skill_prior.dist(hts[:, 0].clone().detach(), detached = True)
            # prior, prior_detach = self.skill_prior.dist(hts[:, 0], detached = True)

        # # -------------- Inverse Dynamics : Skill Learning -------------- #
        # inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts[:,-1])
        inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts_target[:, -1])
        
        if self.cfg.grad_pass.skill:
            skill, invD_skill = batch.skill, inverse_dynamics.rsample()
        else:
            skill = invD_skill = inverse_dynamics.rsample()

        # skill = inverse_dynamics.rsample()

        # skill recon 
        


        # # -------------- Dynamics Learning -------------- #                
        flat_D = self.forward_flatD(hts, skill)
        D = self.forward_D(hts[:, 0], skill)
        
        # only env diff만 
        if self.cfg.manipulation:
            diff_nonPos_latent =  flat_D - hts[:, :-1]
            diff = self.diff_decoder(diff_nonPos_latent.clone().detach())
            diff_target = states[:, 1:, self.cfg.n_pos:] - states[:, :-1, self.cfg.n_pos:]
        else:
            # diff = 0
            # diff_target = 0
            diff_nonPos_latent = flat_D - hts[:, :-1]
            diff = self.diff_decoder(diff_nonPos_latent.clone().detach())
            diff_target = states[:, 1:, self.cfg.n_pos:] - states[:, :-1, self.cfg.n_pos:]


        # # -------------- Subgoal Generator -------------- #
        invD_sub, skill_sub, skill_sub_normal, subgoal_D, subgoal_f, subgoal_f_dist = self.forward_subgoal_G(hts[:, 0], G)


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

            skill_sub = skill_sub,
            skill_sub_normal = skill_sub_normal,
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f,
            # subgoal_target =  subgoal_target,
            subgoal_D_target =  subgoal_f,
            subgoal_f_target =  subgoal_target,
            # subgoal_f_target =  hts[:, -1].clone().detach(),
            subgoal_f_dist = subgoal_f_dist,


            # Difference Decoder for manipulation task 
            diff = diff,
            diff_target = diff_target,
            invD_sub = invD_sub,
        )
        
        # metrics 
        # -------------- Rollout for metric -------------- #
        if not self.training:
            check_subgoals_input = (hts, skill, D, subgoal_f)
            subgoals = self.check_subgoals(check_subgoals_input)
            result.update(subgoals)

        
        return result
    
    # def forward_invD(self, hts):
    def forward_invD(self, start, subgoal):
        # subgoal = subgoal.clone().detach()

        if not self.cfg.grad_pass.invD:
            start = start.clone().detach()
            subgoal = subgoal.clone().detach() 
            
        if self.cfg.noisy_subgoal:
            subgoal += torch.rand_like(subgoal) * 0.001

        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.cfg.tanh)

        return inverse_dynamics, inverse_dynamics_detach
    
    def forward_flatD(self, start, skill):
        # rollout / check 

        if len(start.shape) < 3:
            flat_dynamics_input = torch.cat((start, skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input)

            if self.cfg.diff:
                if self.cfg.grad_pass.D:
                    flat_D = start + flat_D
                else:
                    flat_D = start.clone().detach() + flat_D

        else:
            N, T = start.shape[:2]
            skill_length = T- 1
            skill = skill.unsqueeze(1).repeat(1, skill_length, 1)
            flat_dynamics_input = torch.cat((start[:, :-1], skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)
            
            if self.cfg.diff:
                if self.cfg.grad_pass.D:
                    flat_D = start[:,:-1] + flat_D
                else:
                    flat_D = start.clone().detach()[:,:-1] + flat_D

        return flat_D

    def forward_D(self, start, skill, use_target = False):
        dynamics_input = torch.cat((start, skill), dim=-1)

        if use_target:
            D = self.target_dynamics(dynamics_input)
        else:
            D = self.dynamics(dynamics_input)

        if self.cfg.diff:
            if self.cfg.grad_pass.D:
                D = start + D
            else:
                D = start.clone().detach() + D

        return D
    
    def sg_input(self, start, G):
        return torch.cat((start, G.repeat(1, self.cfg.goal_factor)), dim = -1)

    def forward_subgoal_G(self, start, G):
        start_detached = start.clone().detach() # stop grad : 안하면 goal과의 연관성이 너무 심해짐. 
        sg_input = self.sg_input(start_detached, G)

        if self.cfg.sg_dist:
            subgoal_f_dist = self.subgoal_generator.dist(sg_input)
            subgoal_f = subgoal_f_dist.rsample() + start_detached
            subgoal_f = subgoal_f_dist.rsample() + start_detached

        else:
            _subgoal_f = self.subgoal_generator(sg_input)
            subgoal_f = _subgoal_f + start_detached
            subgoal_f_dist = None

        # invD_sub, _ = self.forward_invD(start = start_detached, subgoal= subgoal_f, use_target= True)
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start_detached, subgoal= subgoal_f, tanh = self.cfg.tanh)

        skill_sub_normal, skill_sub = invD_sub.rsample_with_pre_tanh_value()
        subgoal_D = self.forward_D(start_detached, skill_sub, use_target= True)

        return invD_sub, skill_sub, skill_sub_normal, subgoal_D, subgoal_f, subgoal_f_dist


    @torch.no_grad()
    def check_subgoals(self, inputs):
        hts, skill, D, subgoal_f = inputs
        skill_length = hts.shape[1] - 1
        start = hts[:,0]
        _ht = start.clone()
            
        # 이게 생각보다 많이 차지할텐데.. 지우고 싶다. 
        for _ in range(skill_length):
            _ht = self.forward_flatD(_ht, skill)


        subgoal_recon_D = self.state_decoder(D)
        subgoal_recon_f = self.state_decoder(subgoal_f)

        return edict(
            subgoal_rollout =  _ht,
            subgoal_recon_D = subgoal_recon_D,
            subgoal_recon_f =  subgoal_recon_f,
        )
    
    
    

    def negativeSamples(self, hts, skill):

        with torch.no_grad():
            # 1. D에서 random skill을 수행 -> 대충 noise를 구함
            randomSkill_dist = get_fixed_dist(skill.repeat(1, 2), tanh = self.cfg.tanh)
            random_z_normal, random_z = randomSkill_dist.sample_with_pre_tanh_value()
            # 2. 이걸 실제 state에 추가, negative subgoal을 만든다.
            randomSubgoal = self.forward_D(hts[:, 0], random_z, use_target= True)

        # 3. 그러면 negative sample 완-성. negative sample loss도 계산하면됨. subgoal generator 가이드는 skill dynamics로만 수행함. 따라서 flat은 negative 할 필요 전혀 없음. 
        # 3.a 깔끔하게 하려면 사실 post detach의 크기도 변경하고 그래야되는데 이래저래 구현상 귀찮으므로 별도로 계산 후 0.5를 곱하도록하자. 

        invD_random, _ = self.forward_invD(hts[:,0], randomSubgoal)
        invD_randomSkill = invD_random.rsample()

        # # -------------- Dynamics Learning -------------- #    
        # target을 쓸까?             
        D_random = self.forward_D(hts[:, 0], invD_randomSkill)

        return edict(
            invD_random = invD_random,
            random_z = random_z,
            random_z_normal = random_z_normal,

            D_random = D_random,
            D_random_target = randomSubgoal   
        )

        

    @torch.no_grad()
    def rollout(self, batch):
        states = batch.states
        N, T, _ = states.shape
        skill_length = T - 1
        hts = self.state_encoder(states.view(N * T, -1))
        hts = hts.view(N, T, -1)  

        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()

        if self.cfg.manipulation:
            skill = self.skill_prior.dist(_ht).sample()
        else:
            skill = self.skill_prior.dist(_ht).sample()


        states_rollout = []
        _state = states[:, c]
        skills = [] 

        for i in range(c, self.cfg.plan_H):
            # execute skill on latent space and append to the original sub-trajectory 
            # flat_D로 rollout -> diff in latent space -> diff in raw state space 

            if (i - c) % 5 == 0:
                # skill = self.skill_prior.dist(ht_pos).sample()
                if self.cfg.manipulation:
                    skill = self.skill_prior.dist(_ht).sample()
                else:
                    skill = self.skill_prior.dist(_ht).sample()


            next_ht, cache = self.forward_flatD(_ht, skill) 

            if self.cfg.manipulation:
                diff_raw = self.diff_decoder(next_ht - _ht)
                _state = _state + diff_raw
            else:
                # 여기서는 next_ht가 unseen일까? 아뇨?  
                _state = self.state_decoder(next_ht)
                # _state = pos_raw_state

            states_rollout.append(_state)
            _ht = next_ht
            skills.append(skill)


            
        states_rollout = torch.stack(states_rollout, dim = 1)
        skills = torch.stack(skills, dim = 1)

        result =  edict(
            c = c,
            states_rollout = states_rollout,
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
 
        elif mode == "act":
            # policy or act
            state, G = batch.states, batch.G 
            with torch.no_grad():
                ht = self.state_encoder(state)

            # subgoal 
            # sg_input = torch.cat((ht, G), dim = -1)
            sg_input = self.sg_input(ht, G)
            subgoal_f = self.subgoal_generator(sg_input)
            subgoal_f = subgoal_f + ht

            # invD
            invD, _ = self.forward_invD(ht, subgoal_f)
            skill = invD.rsample() 
            D = self.forward_D(ht, skill)

            result =  edict(
                policy_skill = invD
            )    

            return result 

        else:
            # policy or act
            states, G = batch.states, batch.G
            with torch.no_grad():
                ht, ht_pos, ht_nonPos = self.state_encoder(states)

            # forward subgoal generator 
            sg_input = self.sg_input(ht, G)
            subgoal_f = self.subgoal_generator(sg_input)
            subgoal_f = subgoal_f + ht
            
            # skill inference 
            invD, _ = self.forward_invD(ht, subgoal_f)
            skill = invD.rsample() 
            
            # skill execution
            D = self.forward_D(ht, skill)

            result =  edict(
                policy_skill = invD,
                additional_losses = dict(
                    # GCSL_loss = GCSL_loss
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
        states, G, next_states = batch.states, batch.relabeled_G, batch.next_states

        ht, _, _ = self.state_encoder(states)
        htH,  _, _ = self.state_encoder(next_states)
        htH_target,  _, _ = self.target_state_encoder(next_states)

        # inverse dynamics 
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

        invD_sub, skill_sub, skill_sub_normal, subgoal_D, subgoal_f, subgoal_f_dist = self.forward_subgoal_G(ht, G)
        
        # 실제 subgoal이
        # GCSL_loss = F.mse_loss(subgoal_f, htH_target) + F.mse_loss(subgoal_D, htH_target) + nll_dist(
        #     batch.actions,
        #     invD_sub,
        #     batch.actions_normal,
        #     tanh = self.cfg.tanh
        # ).mean()
        
        weights = torch.softmax(batch.skill_values, dim = 0) * states.shape[0] 

        GCSL_loss_subgoal = weighted_mse(subgoal_f, htH_target, weights) + weighted_mse(subgoal_D, htH_target, weights)
        # GCSL_loss_skill = (nll_dist(
        #     batch.actions,
        #     invD_sub,
        #     batch.actions_normal,
        #     tanh = self.cfg.tanh
        # ) * weights).mean()
        
        # 여기에 kl을 해야 함. 
        GCSL_loss_skill = (nll_dist(
            batch.actions,
            invD_sub,
            batch.actions_normal,
            tanh = self.cfg.tanh
        ) * weights).mean()


        # 이렇게하면 mode covering임.
        # mode dropping ㄱ 

        GCSL_loss = GCSL_loss_subgoal + GCSL_loss_skill

        
        return  edict(
            state_consistency = state_consistency,
            skill_consistency = skill_consistency,
            GCSL_loss = GCSL_loss
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
                "f" : {
                    "params" :  self.subgoal_generator.parameters(), 
                    "lr" : self.cfg.f_lr, 
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
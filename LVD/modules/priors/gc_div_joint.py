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


class GoalConditioned_Diversity_Joint_Prior(ContextPolicyMixin, BaseModule):
    """
    """
    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_state_encoder = deepcopy(self.state_encoder)
        self.target_state_decoder = deepcopy(self.state_decoder)
        self.target_inverse_dynamics = deepcopy(self.inverse_dynamics)
        self.target_dynamics = deepcopy(self.dynamics)
        self.target_flat_dynamics = deepcopy(self.flat_dynamics)

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
        update_moving_average(self.target_state_decoder, self.state_decoder, 1)
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)
        self.n_soft_update += 1

    def forward(self, batch, *args, **kwargs):
        states, G = batch.states, batch.G
        N, T, _ = states.shape
        skill_length = T - 1 


        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr = self.state_encoder(states.view(N * T, -1)) # N * T, -1 
        states_hat = self.state_decoder(states_repr).view(N, T, -1) 
        hts = states_repr.view(N, T, -1).clone()

        with torch.no_grad():
            ppc_states = self.state_processor.to_ppc(states.clone())  # env states removed 
            ppc = self.state_encoder(ppc_states.view(N * T, -1)) # forward 

            # # for MMD loss of WAE
            state_emb = states_repr.view(N, T, -1)[:, 0] # N, -1 
            states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda() # 512, -1 

            # target for dynamics & subgoal generator 
            hts_target = self.target_state_encoder(states.view(N * T, -1))
            states_hat_target = self.state_decoder(hts_target).view(N, T, -1)
            
            hts_target = hts_target.view(N, T, -1)
            _hts = hts.clone().detach()
            start, subgoal = _hts[:, 0], _hts[:, -1]


            # hts_target = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)
            subgoal_target = hts_target[:, -1]

        # -------------- State-Conditioned Prior -------------- #
        states_ppc_pred = self.state_to_ppc(states_repr.clone().detach())
        prior, prior_detach = self.skill_prior.dist(ppc.view(N,T,-1)[:, 0], detached = True)

        # # -------------- Inverse Dynamics : Skill Learning -------------- #
        # inverse_dynamics, inverse_dynamics_detach = self.forward_invD(hts[:,0], hts[:,-1])
        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.tanh)
        skill = inverse_dynamics.rsample()

        # # -------------- Dynamics Learning -------------- #                
        # flat_D = self.forward_flatD(hts, skill)
        flat_dynamics_input = torch.cat((hts[:, :-1], skill.unsqueeze(1).repeat(1, skill_length, 1)), dim=-1)
        diff_flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)
        flat_D =  hts[:,:-1] + diff_flat_D

        # D = self.forward_D(hts[:, 0], skill)
        dynamics_input = torch.cat((hts[:, 0], skill), dim = -1)
        diff_D = self.dynamics(dynamics_input)
        D =  hts[:,0] + diff_D

        # # -------------- Subgoal Generator -------------- #
        # invD_sub, subgoal_D, subgoal_f = self.forward_subgoal_G(hts[:, 0], G)
        sg_input = torch.cat((start,  G), dim = -1)
        diff_subgoal_f = self.subgoal_generator(sg_input)
        subgoal_f = diff_subgoal_f + start
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start, subgoal = subgoal_f, tanh = self.tanh)

        skill_sub = invD_sub.rsample()


        dynamics_input = torch.cat((start, skill_sub), dim = -1)
        diff_subgoal_D = self.target_dynamics(dynamics_input)
        subgoal_D = start + diff_subgoal_D 




        # -------------- Rollout for metric -------------- #
        check_subgoals_input = (hts, inverse_dynamics, D, subgoal_D, subgoal_f)
        subgoals = self.check_subgoals(check_subgoals_input)

        result = edict(
            # states
            states = states,
            states_repr = states_repr,
            # states_repr = hts[:, 0], # 너무 오래걸려서 줄임. 
            hts = hts,
            states_hat_target = states_hat_target,
            states_hat = states_hat,
            states_fixed_dist = states_fixed,
            states_ppc = states_ppc_pred.view(N, T, -1),
            states_ppc_target = ppc.view(N, T, -1),
    
            prior = prior,
            prior_detach = prior_detach,
            invD = inverse_dynamics,
            invD_detach = inverse_dynamics_detach,
            # Ds
            D = D,
            flat_D = flat_D,
            # D_target =  hts[:,-1].clone().detach(), 
            # flat_D_target = hts[:, 1:].clone().detach(),

            D_target =  subgoal_target, 
            flat_D_target = hts_target[:, 1:],

            # f
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f,
            # "subgoal_target" : subgoal,
            subgoal_target =  subgoal_target,

            invD_sub = invD_sub,
            # "invD_sub2" : invD_sub2,
        )

        result.update(subgoals)
        return result
    
    # def forward_invD(self, hts):
    def forward_invD(self, start, subgoal):
        # if not self.grad_pass_invD:
        #     hts = hts.clone().detach()
        # start, subgoal = hts[:,0], hts[:, -1]
        if not self.grad_pass_invD:
            start = start.clone().detach()
            subgoal = subgoal.clone().detach()

        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.tanh)
        return inverse_dynamics, inverse_dynamics_detach
    
    def forward_flatD(self, start, skill):
        # rollout / check 
        if len(start.shape) < 3:
            flat_dynamics_input = torch.cat((start, skill), dim=-1)
            flat_D = self.target_flat_dynamics(flat_dynamics_input)
            if self.diff:
                if self.grad_pass_D:
                    # flat_D += start
                    flat_D = start + flat_D

                else:
                    flat_D = start.clone().detach() + flat_D
                    # flat_D += start.clone().detach()
        else:
            N, T = start.shape[:2]
            skill_length = T- 1
            skill = skill.unsqueeze(1).repeat(1, skill_length, 1)
            flat_dynamics_input = torch.cat((start[:, :-1], skill), dim=-1)
            flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)

            if self.diff:
                if self.grad_pass_D:
                    # flat_D += start[:,:-1]
                    flat_D = start[:,:-1] + flat_D

                else:
                    # flat_D += start[:,:-1].clone().detach()
                    flat_D = start[:,:-1].clone().detach() + flat_D

        return flat_D

    def forward_D(self, start, skill, use_target = False):

        dynamics_input = torch.cat((start, skill), dim = -1)
        if use_target:
            D = self.target_dynamics(dynamics_input)
        else:
            D = self.dynamics(dynamics_input)

        if self.diff:
            if self.grad_pass_D:
                # D += start
                D = start + D
            else:
                # D += start.clone().detach()
                D = start.clone().detach() + D
        return D

    def forward_subgoal_G(self, start, G):
        start = start.clone().detach() # stop grad : 안하면 goal과의 연관성이 너무 심해짐. 

        sg_input = torch.cat((start,  G), dim = -1)
        subgoal_f = self.subgoal_generator(sg_input)
        if self.diff:
            subgoal_f = subgoal_f + start
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start, subgoal = subgoal_f, tanh = self.tanh)
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
            _ht = self.forward_flatD(_ht, skill)
        invD_rollout_main, _ = self.target_inverse_dynamics.dist(state = start, subgoal = _ht, tanh = self.tanh)

        subgoal_recon = self.target_state_decoder(subgoal)
        subgoal_recon_D = self.target_state_decoder(D)
        subgoal_recon_D_f = self.target_state_decoder(subgoal_D)
        subgoal_recon_f = self.target_state_decoder(subgoal_f)

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

        hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)  

        hts_rollout = []
        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()
        skill_sampled_orig = self.skill_prior.dist(self.state_to_ppc(_ht)).sample()
        skill_sampled = skill_sampled_orig.clone()
        # 1 skill
        for _ in range(c, skill_length):
            # execute skill on latent space and append to the original sub-trajectory 
            _ht = self.forward_flatD(_ht, skill_sampled)
            hts_rollout.append(_ht)
        
        # # for f learning, execute 4 skill more
        # for _ in range(9):
        #     skill = self.skill_prior.dist(self.state_to_ppc(_ht)).sample()
        #     _ht = self.forward_D(_ht, skill)
        #     hts_rollout.append(_ht)
            
        # hts_rollout = torch.stack(hts_rollout, dim = 1)
        # N, T, _ = hts_rollout.shape
        # states_rollout = self.target_state_decoder( hts_rollout.view(N * T, -1), rollout = True).view(N, T, -1)

        # result =  edict(
        #     c = c,
        #     states_rollout = states_rollout,
        #     skill_sampled = skill_sampled_orig,
        #     # hts_rollout = hts_rollout
        # )
        
        # for f learning, execute 4 skill more
        for i in range(90):
            if i % 10 == 0:
                skill = self.skill_prior.dist(self.state_to_ppc(_ht)).sample()
            _ht = self.forward_flatD(_ht, skill)
            hts_rollout.append(_ht)
            
        hts_rollout = torch.stack(hts_rollout, dim = 1)
        N, T, _ = hts_rollout.shape
        states_rollout = self.target_state_decoder( hts_rollout.view(N * T, -1), rollout = True).view(N, T, -1)

        result =  edict(
            c = c,
            states_rollout = states_rollout,
            skill_sampled = skill_sampled_orig,
            # hts_rollout = hts_rollout
        )


        return result 

    def encode(self, states, keep_grad = False, prior = False):
        if keep_grad:
            if prior:
                ht = self.state_to_ppc(self.state_encoder(states))
            else:
                ht = self.state_encoder(states)

        else:
            with torch.no_grad():
                if prior:
                    ht = self.state_to_ppc(self.state_encoder(states))
                else:
                    ht = self.state_encoder(states)
        return ht

    def dist(self, batch, mode = "policy"):
        assert mode in ['policy', 'consistency', 'act'], "Invalid mode"
        if mode == "consistency":
            return self.consistency(batch)
        else:
            # policy or act
            state, G = batch.states, batch.G 
            with torch.no_grad():
                ht = self.state_encoder(state)

            # subgoal 
            _, _, subgoal_f = self.forward_subgoal_G(ht, G)

            inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)
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

        states, G, next_states = batch.states, batch.G, batch.next_states
        # self.state_encoder.eval()
        ht = self.state_encoder(states)
        htH = self.state_encoder(next_states)
        htH_target = self.target_state_encoder(next_states)


        # inverse dynamics 
        if self.grad_pass_invD:
            invD, invD_detach = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.tanh)
        else:
            invD, invD_detach = self.inverse_dynamics.dist(state = ht.clone().detach(), subgoal = htH.clone().detach(), tanh = self.tanh)
        D = self.forward_D(ht, batch.actions)

        # state_consistency = F.mse_loss(ht.clone().detach() + diff, htH) + mmd_loss
        state_consistency = F.mse_loss(D, htH_target) #+ mmd_loss
        skill_consistency = nll_dist(
            batch.actions,
            invD,
            batch.actions_normal,
            tanh = self.tanh
        ).mean()
        
        # GCSL
        # 똑같은 skill 뽑으면 됨./ 

        invD_sub, subgoal_D, subgoal_f = self.forward_subgoal_G(ht, G)

        # GCSL_loss = F.mse_loss(diff_subgoal_f, diff_subgoal_D) + nll_dist(
        #     batch.actions,
        #     invD_sub,
        #     batch.actions_normal,
        #     tanh = self.tanh
        # ).mean()
        
        # GCSL_loss = F.mse_loss(diff_subgoal_f + ht, htH) + torch_dist.kl_divergence(invD_sub, invD_detach).mean()
        GCSL_loss = F.mse_loss(subgoal_f, htH_target) + torch_dist.kl_divergence(invD_sub, invD_detach).mean()



        return  edict(
            state_consistency = state_consistency,
            skill_consistency = skill_consistency,
            GCSL_loss = GCSL_loss.item()
            # GCSL_loss = GCSL_loss
        )


    def get_rl_params(self):
        return edict(
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
                # "f" : {
                #     "params" :  self.subgoal_generator.parameters(), 
                #     "lr" : self.cfg.f_lr, 
                #     # "metric" : "GCSL_loss"
                #     "metric" : None,
                # }


            }
                
        )

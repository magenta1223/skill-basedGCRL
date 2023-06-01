import torch
import torch.distributions as torch_dist
import copy
from torch.nn import functional as F
import math
import random
from ...contrib import TanhNormal
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import update_moving_average
from easydict import EasyDict as edict

import d4rl

class GoalConditioned_Diversity_Joint_Prior(ContextPolicyMixin, BaseModule):
    """
    """
    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.target_state_decoder = copy.deepcopy(self.state_decoder)
        self.target_inverse_dynamics = copy.deepcopy(self.inverse_dynamics)
        self.target_dynamics = copy.deepcopy(self.dynamics)
        self.target_flat_dynamics = copy.deepcopy(self.flat_dynamics)

        self.n_soft_update = 1
        self.update_freq = 5

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # soft update
        if self.n_soft_update % self.update_freq == 0:
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
        states_repr = self.state_encoder(states.view(N * T, -1))
        states_hat = self.state_decoder(states_repr).view(N, T, -1)
        hts = states_repr.view(N, T, -1).clone().detach()

        with torch.no_grad():
            # for MMD loss of WAE
            state_emb = states_repr.view(N, T, -1)[:, 0]
            states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda()

            # inverse dynamics, skill prior input 
            _hts = hts.clone().detach()
            start, subgoal = _hts[:, 0], _hts[:, -1]
            # target for dynamics & subgoal generator 
            hts_target = self.target_state_encoder(states.view(N * T, -1)).view(N, T, -1)
            subgoal_target = hts_target[:, -1]


        # states_repr = self.state_encoder(states.view(N * T, -1))
        # states_hat = self.state_decoder(states_repr).view(N, T, -1)
        # hts = states_repr.view(N, T, -1).clone()
        # start, subgoal = hts[:, 0], hts[:, -1]
        # with torch.no_grad():
        #     # for MMD loss of WAE
        #     state_emb = states_repr.view(N, T, -1)[:, 0]
        #     states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda()

        #     # target for dynamics & subgoal generator 
        #     hts_target = self.target_state_encoder(states.view(N * T, -1)).view(N, T, -1)
        #     subgoal_target = hts_target[:, -1]






        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.skill_prior.dist(start, detached = True)

        # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.tanh)
        
        # -------------- Dynamics Learning -------------- #        
        skill = inverse_dynamics.rsample()
        
        # flat dynamcis for rollout
        flat_dynamics_input = torch.cat((hts[:, :-1], skill.unsqueeze(1).repeat(1, skill_length, 1)), dim=-1)
        diff_flat_D = self.flat_dynamics(flat_dynamics_input.view(N * skill_length, -1)).view(N, skill_length, -1)
        flat_D =  hts_target[:,:-1] + diff_flat_D

        # skill dynamcis for regularization
        dynamics_input = torch.cat((hts[:, 0], skill), dim = -1)
        diff_D = self.dynamics(dynamics_input)
        D = hts_target[:, 0].clone().detach() + diff_D

        # -------------- Subgoal Generator -------------- #
        sg_input = torch.cat((start,  G), dim = -1)
        diff_subgoal_f = self.subgoal_generator(sg_input)
        subgoal_f = diff_subgoal_f + start
        invD_sub, _ = self.target_inverse_dynamics.dist(state = start, subgoal = subgoal_f, tanh = self.tanh)

        skill_sub = invD_sub.rsample()

        dynamics_input = torch.cat((start, skill_sub), dim = -1)
        diff_subgoal_D = self.target_dynamics(dynamics_input)
        subgoal_D = start + diff_subgoal_D 

        # -------------- Rollout for metric -------------- #
        _ht = start.clone().detach()
        # dense execution with loop (for metric)
        with torch.no_grad():
            for _ in range(skill_length):
                flat_dynamics_input = torch.cat((_ht, skill), dim=-1)
                diff = self.target_flat_dynamics(flat_dynamics_input) 
                _ht = _ht + diff
            invD_rollout_main, _ = self.target_inverse_dynamics.dist(state = start, subgoal = _ht, tanh = self.tanh)

            subgoal_recon = self.target_state_decoder(subgoal)
            subgoal_recon_D = self.target_state_decoder(D)
            subgoal_recon_D_f = self.target_state_decoder(subgoal_D)
            subgoal_recon_f = self.target_state_decoder(subgoal_f)


        result = edict(
            # states
            states = states,
            states_repr = state_emb,
            hts = hts,
            states_hat = states_hat,
            states_fixed_dist = states_fixed,
            prior = prior,
            prior_detach = prior_detach,
            invD = inverse_dynamics,
            invD_detach = inverse_dynamics_detach,
            # Ds
            D = D,
            flat_D = flat_D,
            D_target =  subgoal_target, 
            flat_D_target = hts_target[:, 1:],

            # f
            subgoal_D =  subgoal_D,
            subgoal_f = subgoal_f,
            # "subgoal_target" : subgoal,
            subgoal_target =  subgoal_target,

            invD_sub = invD_sub,
            z_sub  = skill_sub,
            # "invD_sub2" : invD_sub2,

            # for metric
            z_invD = skill,
            invD_rollout_main= invD_rollout_main,
            subgoal_rollout =  _ht,
            subgoal_recon_D = subgoal_recon_D,
            subgoal_recon_f =  subgoal_recon_f,
            subgoal_recon_D_f =  subgoal_recon_D_f,
            subgoal_recon =  subgoal_recon

        )

        if self.skill_prior_ppc is not None:
            result['prior_ppc'] = self.skill_prior_ppc.dist(states[:, 0])

        return result
    

    @torch.no_grad()
    def rollout(self, batch):
        self.state_encoder.eval()
        self.state_decoder.eval()
        self.skill_prior.eval()
        self.inverse_dynamics.eval()
        self.flat_dynamics.eval()
        self.dynamics.eval()
        
        if self.skill_prior_ppc is not None:
            self.skill_prior_ppc.eval()

        states, skill = batch.states, batch.G
        N, T, _ = states.shape
        skill_length = T - 1

        hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)  

        hts_rollout = []
        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()

        if self.skill_prior_ppc is not None:
            # env state agnostic
            skill_sampled_orig = self.skill_prior_ppc.dist(states[:, 0]).sample()
        else:
            skill_sampled_orig = self.skill_prior.dist(_ht).sample()
    

        skill_sampled = skill_sampled_orig.clone()
        # 1 skill
        for _ in range(c, skill_length):
            # execute skill on latent space and append to the original sub-trajectory 
            dynamics_input = torch.cat((_ht, skill_sampled), dim=-1)
            diff = self.flat_dynamics(dynamics_input) 
            _ht = _ht + diff
            hts_rollout.append(_ht)
        
        invD_rollout, _ = self.inverse_dynamics.dist(state = hts[:, 0], subgoal = _ht,  tanh = self.tanh)
        invD_GT, _ = self.inverse_dynamics.dist(state = hts[:, 0], subgoal = hts[:, -1],  tanh = self.tanh)


        # for f learning, execute 4 skill more
        for _ in range(9):
            skill = self.skill_prior.dist(_ht).sample()
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            diff = self.dynamics(dynamics_input) 
            _ht = _ht + diff
            hts_rollout.append(_ht)

            
        hts_rollout = torch.stack(hts_rollout, dim = 1)
        N, T, _ = hts_rollout.shape
        states_rollout = self.target_state_decoder( hts_rollout.view(N * T, -1), rollout = True).view(N, T, -1)

        result =  edict(
            c = c,
            states_rollout = states_rollout,
            skill_sampled = skill_sampled_orig,
            invD_rollout = invD_rollout,
            invD_GT = invD_GT,
            hts_rollout = hts_rollout
        )
        
        # {
        #     "c" : c,
        #     "states_rollout" : states_rollout,
        #     "skill_sampled" : skill_sampled_orig,
        #     "invD_rollout" : invD_rollout,
        #     "invD_GT" : invD_GT,
        #     "hts_rollout" : hts_rollout 
        # }
        return result 

    def encode(self, states, keep_grad = False):
        if keep_grad:
            ht = self.state_encoder(states)
        else:
            with torch.no_grad():
                ht = self.state_encoder(states)
        return ht

    def dist(self, batch, mode = "policy"):
        assert mode in ['policy', 'consistency', 'act'], "Invalid mode"
        if mode == "consistency":
            return self.consistency(batch)
        else:
            state, G = batch.states, batch.G 

            with torch.no_grad():
                ht = self.state_encoder(state)

            # subgoal 
            sg_input = torch.cat((ht,  G), dim = -1)
            diff_subgoal_f = self.subgoal_generator(sg_input)
            subgoal_f = diff_subgoal_f + ht

            inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)
            # inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)


            skill = inverse_dynamics_hat.rsample() 
            dynamics_input = torch.cat((ht,  skill), dim = -1)
            # subgoal_D = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
            diff = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
            
            result =  edict(
                policy_skill = inverse_dynamics_hat,
                additional_losses = dict(
                    state_consistency_f = F.mse_loss(diff_subgoal_f, diff)
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
        self.state_encoder.eval()
        ht = self.state_encoder(states)
        htH = self.target_state_encoder(next_states)
        
        # inverse dynamics 
        invD, invD_detach = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.tanh)
        z = invD.rsample()
                
        # dynamics 
        dynamics_input = torch.cat((ht,  z), dim = -1)
        diff = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 

        state_consistency = F.mse_loss(ht + diff, htH)
        skill_consistency = nll_dist(
            batch.actions,
            invD,
            batch.actions_normal,
            tanh = self.tanh
        ).mean()
        
        # GCSL
        # 똑같은 skill 뽑으면 됨./ 


        sg_input = torch.cat((ht,  G), dim = -1)
        diff_subgoal_f = self.subgoal_generator(sg_input)
        subgoal_f = diff_subgoal_f + ht
        invD_sub, _ = self.target_inverse_dynamics.dist(state = ht, subgoal = subgoal_f, tanh = self.tanh)

        skill_sub = invD_sub.rsample()

        dynamics_input = torch.cat((ht, skill_sub), dim = -1)
        diff_subgoal_D = self.target_dynamics(dynamics_input)
        subgoal_D = ht + diff_subgoal_D 

        # GCSL_loss = F.mse_loss(diff_subgoal_f, diff_subgoal_D) + nll_dist(
        #     batch.actions,
        #     invD_sub,
        #     batch.actions_normal,
        #     tanh = self.tanh
        # ).mean()
        
        GCSL_loss = F.mse_loss(diff_subgoal_f + ht, htH) + torch_dist.kl_divergence(invD_sub, invD_detach).mean()


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
                # "f" : {
                #     "params" :  self.subgoal_generator.parameters(), 
                #     "lr" : self.cfg.f_lr, 
                #     # "metric" : "GCSL_loss"
                #     "metric" : None,
                # }

            }
                
        )

import torch
import copy

import math

from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class GoalConditioned_Diversity_Joint_Prior(BaseModule):
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

    def forward(self, inputs, *args, **kwargs):
        states, G = inputs['states'], inputs['G']  
        N, T, _ = states.shape
        skill_length = T - 1 

        # -------------- State Enc / Dec -------------- #
        # for state reconstruction and dynamics
        states_repr = self.state_encoder(states.view(N * T, -1))
        states_hat = self.state_decoder(states_repr).view(N, T, -1)
        hts = states_repr.view(N, T, -1).clone()

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

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.prior_policy.dist(start, detached = True)

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
            # invD
            "invD" : inverse_dynamics,
            "invD_detach" : inverse_dynamics_detach,
            # Ds
            "D" : D,
            "flat_D" : flat_D,

            "D_target" : subgoal_target, 
            "flat_D_target" : hts_target[:, 1:],

            # f
            # "subgoal_D" : subgoal_D.clone().detach(),
            "subgoal_D" : subgoal_D,
            "subgoal_f" : subgoal_f,
            # "subgoal_target" : subgoal,
            "subgoal_target" : subgoal_target,

            "invD_sub" : invD_sub,
            "z_sub" : skill_sub,
            # "invD_sub2" : invD_sub2,



            # for metric
            "z_invD" : skill,
            "invD_rollout_main" : invD_rollout_main,
            "subgoal_rollout" : _ht,
            "subgoal_recon_D" : subgoal_recon_D,
            "subgoal_recon_f" : subgoal_recon_f,
            "subgoal_recon_D_f" : subgoal_recon_D_f,
            "subgoal_recon"  : subgoal_recon

            # "D_metric" : D,
            # "D_target_metric" : subgoal, 
            # "flat_D" : flat_D,
            # "flat_D_target" : hts[:, 1:],
        }

        if self.prior_proprioceptive is not None:
            result['prior_ppc'] = self.prior_proprioceptive.dist(states[:, 0])

        return result
    

    @torch.no_grad()
    def rollout(self, inputs):
        self.state_encoder.eval()
        self.state_decoder.eval()
        self.prior_policy.eval()
        self.inverse_dynamics.eval()
        self.flat_dynamics.eval()
        self.dynamics.eval()
        
        if self.prior_proprioceptive is not None:
            self.prior_proprioceptive.eval()

        states, skill = inputs['states'], inputs['actions']
        N, T, _ = states.shape
        skill_length = T - 1

        hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)  

        hts_rollout = []
        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()

        if self.prior_proprioceptive is not None:
            # env state agnostic
            skill_sampled_orig = self.prior_proprioceptive.dist(states[:, 0]).sample()
        else:
            skill_sampled_orig = self.prior_policy.dist(_ht).sample()
    

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
            skill = self.prior_policy.dist(_ht).sample()
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            diff = self.dynamics(dynamics_input) 
            _ht = _ht + diff
            hts_rollout.append(_ht)

            
        hts_rollout = torch.stack(hts_rollout, dim = 1)
        N, T, _ = hts_rollout.shape
        states_rollout = self.target_state_decoder( hts_rollout.view(N * T, -1), rollout = True).view(N, T, -1)

        result =  {
            "c" : c,
            "states_rollout" : states_rollout,
            "skill_sampled" : skill_sampled_orig,
            "invD_rollout" : invD_rollout,
            "invD_GT" : invD_GT,
            "hts_rollout" : hts_rollout 
        }
        return result 



    def dist(self, inputs):
        """
        """
        state, G = inputs['states'], inputs['G']       

        with torch.no_grad():
            # ht, G = self.state_encoder(state), self.state_encoder(G)
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


        result = {
            "inverse_D" : inverse_dynamics_hat,
            "subgoal" : diff + ht,
            "subgoal_target" : subgoal_f,
        }

        return result
    
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 

        dist_inputs = dict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(dist_inputs, "eval")['inverse_D']
        return dist.sample().detach().cpu().squeeze(0).numpy()
        # # TODO explore 여부에 따라 mu or sample을 결정
        # if self.prior_policy.tanh:
        #     z_normal, z = dist.rsample_with_pre_tanh_value()
        #     # to calculate kld analytically 
        #     loc, scale = dist._normal.base_dist.loc, dist._normal.base_dist.scale 

        #     return z_normal.detach().cpu().squeeze(0).numpy(), z.detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        # else:
        #     loc, scale = dist.base_dist.loc, dist.base_dist.scale
        #     return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()

    def finetune(self, inputs):
        """
        Finetune inverse dynamics and dynamics with the data collected in online.
        """

        states, G, next_states = inputs['states'], inputs['G'], inputs['next_states'] 

        with torch.no_grad():
            ht = self.state_encoder(states)
            htH = self.state_encoder(next_states)

        # ht = self.state_encoder(states)
        # htH = self.target_state_encoder(next_states)

        # finetune invD, D 
        invD, _ = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.tanh)
        z = invD.rsample()
                
        result = {
            "inverse_D" : invD,
            "subgoal_target" : htH
        }

        dynamics_input = torch.cat((ht,  z), dim = -1)
        diff = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
        result['subgoal'] = ht + diff

        return result



    def __prior__(self, inputs):
        states = inputs['states']

        # with torch.no_grad():
        #     ht = self.state_encoder(states)
        #     prior = self.prior_policy.dist(ht)
        with torch.no_grad():
            ht = self.state_encoder(states)
            # if self.env_name == "maze":
            #     # only visual embedding 
            #     prior = self.prior_policy.dist(ht[:, :32])

            # else:
            prior = self.prior_policy.dist(ht)

        return {
            "prior" : prior
        }
import torch
from torch.nn import functional as F
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import TanhNormal
from easydict import EasyDict as edict
import d4rl


class SPiRL_Prior(ContextPolicyMixin, BaseModule):
    """
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

    def forward(self, batch, *args, **kwargs):
        """
        """
        states, G = batch.states, batch.G
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        prior = self.forward_prior(states) 
        if self.tanh:
            prior_dist = prior._normal.base_dist
        else:
            prior_dist = prior.base_dist
        prior_locs, prior_scales = prior_dist.loc.clone().detach(), prior_dist.scale.clone().detach()
        prior_pre_scales = inverse_softplus(prior_scales)

        res_locs, res_pre_scales = self.highlevel_policy(torch.cat((states[:,0], G), dim = -1)).chunk(2, dim=-1)

        locs = res_locs + prior_locs
        scales = F.softplus(res_pre_scales + prior_pre_scales)
        policy_skill = get_dist(locs, scale = scales, tanh = self.tanh)

        return edict(
            prior = prior,
            states = states,
            policy_skill = policy_skill,
        )

    def forward_prior(self, states):
        if len(states.shape) > 2:
            states = states[:, 0]
        
        if self.cfg.manipulation:
            states = states[:, :self.cfg.n_pos]
            
        return self.skill_prior.dist(states)


    def encode(self, states, keep_grad = False, prior = False):
        if self.cfg.manipulation:
            return states[..., :self.cfg.n_pos]
        else:
            return states

    def soft_update(self):
        pass

        
    def dist(self, batch, mode = "policy"):
        """
        """
        assert mode in ['policy', 'consistency', 'act'], "Invalid mode"

        if mode == "consistency":
            return self.consistency(batch)
        else:
            states, G = batch.states, batch.G
            with torch.no_grad():
                prior = self.forward_prior(states)

                if self.tanh:
                    prior_dist = prior._normal.base_dist
                else:
                    prior_dist = prior.base_dist
                    
                prior_locs, prior_scales = prior_dist.loc.clone().detach(), prior_dist.scale.clone().detach()
                prior_pre_scales = inverse_softplus(prior_scales)
                
            res_locs, res_pre_scales = self.highlevel_policy(torch.cat((states, G), dim = -1)).chunk(2, dim=-1)

            locs = res_locs + prior_locs
            scales = F.softplus(res_pre_scales + prior_pre_scales)
            policy_skill = get_dist(locs, scale = scales, tanh = self.tanh)

            return edict(
                prior = prior,
                policy_skill = policy_skill,
                additional_losses  = {}
            )

    @torch.no_grad()
    def act(self, states, G):
        dist_inputs = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )
        dist = self.dist(dist_inputs).policy_skill

        if isinstance(dist, TanhNormal):
            if self.explore:
                z_normal, z = dist.sample_with_pre_tanh_value()
                return to_skill_embedding(z_normal), to_skill_embedding(z)
            else: 
                return to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2)), to_skill_embedding((torch.tanh(dist._normal.base_dist.loc) * 2))

        else:
            return None, to_skill_embedding(dist.sample())
        

    def consistency(self, batch):
        states, G = batch.states, batch.G
        with torch.no_grad():
            prior = self.forward_prior(states)
            if self.tanh:
                prior_dist = prior._normal.base_dist
            else:
                prior_dist = prior.base_dist
            prior_locs, prior_scales = prior_dist.loc.clone().detach(), prior_dist.scale.clone().detach()
            prior_pre_scales = inverse_softplus(prior_scales)
            
        res_locs, res_pre_scales = self.highlevel_policy(torch.cat((states, G), dim = -1)).chunk(2, dim=-1)

        locs = res_locs + prior_locs
        scales = F.softplus(res_pre_scales + prior_pre_scales)
        policy_skill = get_dist(locs, scale = scales, tanh = self.tanh)

        skill_consistency = nll_dist(
            batch.actions,
            policy_skill,
            batch.actions_normal,
            tanh = self.tanh
        ).mean() # gcsl loss 

        return edict(
            skill_consistency = skill_consistency
        )

    def get_rl_params(self):
        

        return edict(
            policy = [
                {"params" : self.highlevel_policy.parameters(), "lr" : self.cfg.policy_lr}
            ],
            consistency = {
                "highpolicy" : {"params" : self.highlevel_policy.parameters(), "lr" : self.cfg.gcsl_lr, "metric" : None},
            }
        )

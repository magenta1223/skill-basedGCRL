import torch
from torch.nn import functional as F
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import TanhNormal
from easydict import EasyDict as edict
import d4rl

class Flat_GCSL(ContextPolicyMixin, BaseModule):
    """
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

    def forward(self, batch, *args, **kwargs):
        """
        """
        states, G = batch.states, batch.G
        
        # -------------- State Enc / Dec -------------- #

        policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
        policy_action = policy_action_dist.rsample()
        
        return edict(
            states = states,
            policy_action_dist = policy_action_dist,
            policy_action = policy_action,
        )

    def soft_update(self):
        pass

    def encode(self, states, keep_grad = False):
        return states

    def dist(self, batch, mode = "policy"):
        """
        """
        if mode == "consistency":
            # G is always relabeled in GCSL.
            # See collector.common.GC_Buffer_Relabel.ep_to_transitions()
            states, G = batch.states, batch.G


            policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
            policy_action = policy_action_dist.rsample()

            skill_consistency = F.mse_loss(policy_action, batch.actions)

            return edict(
                skill_consistency = skill_consistency
            )

        else:
            # act 
            states, G = batch.states, batch.G    

            policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))

            return edict(
                policy_skill = policy_action_dist
            )

    @torch.no_grad()
    def act(self, states, G):
        dist_inputs = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(dist_inputs).policy_skill
        if isinstance(dist, TanhNormal):
            z_normal, z = dist.sample_with_pre_tanh_value()
            return to_skill_embedding(z_normal), to_skill_embedding(z)
        else:
            return None, to_skill_embedding(dist.sample())

    def get_rl_params(self):
        
        return edict(
            policy = [
                {"params" : self.policy.parameters(), "lr" : self.cfg.policy_lr}
            ],
            consistency = {
                "policy" : {
                    "params" : self.policy.parameters(),
                    "lr" : self.cfg.gcsl_lr, 
                    "metric" : None,
                    # "metric" : "skill_consistency"
                    },
            }
        )

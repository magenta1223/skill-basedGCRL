import torch
from torch.nn import functional as F
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import TanhNormal
from easydict import EasyDict as edict
from copy import deepcopy




class RIS_Prior(ContextPolicyMixin, BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

        self.prior_policy = deepcopy(self.policy)
        # self.high_level_policy = deepcopy(self.policy)


    def forward(self, batch, mode = "default" , *args, **kwargs):
        """
        State only Conditioned Prior
        inputs : dictionary
            -  states 
        return: state conditioned prior, and detached version for metric
        """
        states, G = batch.states, batch.G
        


        # -------------- State Enc / Dec -------------- #
        # N, T = states.shape[:2]
        # states_unrolled = states.view(N * T, -1) # N * T, -1
        # G_unrolled = G.unsqueeze(1).repeat(1, T, 1).view(N * T, -1)
        # policy_input = torch.cat((states_unrolled, G_unrolled), dim = -1)
        # policy_skill = self.policy(policy_input).view(N, T, -1)
        # policy_skill = torch.tanh(policy_skill)

        # policy_skill = self.policy(torch.cat((states, G), dim = -1))
        # policy_skill = torch.tanh(policy_skill)
        # tanh normal로 ? 
        # 혼합

        if mode == "default":

            policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
            policy_action = policy_action_dist.rsample()

            subgoal_dist = self.high_level_policy.dist( torch.cat((states, G), dim = -1))
            subgoal = subgoal_dist.sample()
            subgoal_action_dist = self.prior_policy.dist(torch.cat((states, subgoal), dim = -1))

            return edict(
                states = states,
                policy_action_dist = policy_action_dist,
                policy_action = policy_action,
                subgoal_action_dist = subgoal_action_dist,
            )



        else:

            subgoal_dist = self.high_level_policy.dist( torch.cat((states, G), dim = -1))
            return edict(
                subgoal_dist = subgoal_dist,
            )



    def soft_update(self):
        pass

    def encode(self, states, keep_grad = False):
        return states

    def dist(self, batch, mode = "policy"): # 이제 이게 필요가 없음. 
        """
        """
        if mode == "consistency":
            states, G = batch.states, batch.relabeled_G


            policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
            policy_action = policy_action_dist.rsample()

            # policy_skill = self.policy(torch.cat((states, G), dim = -1))
            # policy_skill = torch.tanh(policy_skill)
            skill_consistency = F.mse_loss(policy_action, batch.actions)

            return edict(
                skill_consistency = skill_consistency
            )

        else:
            states, G = batch.states, batch.G    
            # policy_skill = self.policy(torch.cat((states, G), dim = -1))
            # policy_skill = torch.tanh(policy_skill)

            policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
            # policy_skill = policy_action_dist.rsample()

            return edict(
                policy_skill = policy_action_dist
            )

    @torch.no_grad()
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
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
                    "lr" : self.cfg.consistency_lr, 
                    "metric" : None,
                    # "metric" : "skill_consistency"
                    },
            }
        )

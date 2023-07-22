import torch
from torch.nn import functional as F
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import TanhNormal
from easydict import EasyDict as edict
import d4rl

class Flat_GCSL(ContextPolicyMixin, BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

    def forward(self, batch, *args, **kwargs):
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
        
        policy_action_dist = self.policy.dist(torch.cat((states, G), dim = -1))
        z = policy_action_dist.rsample()

        

        return edict(
            states = states,
            policy_action_dist = policy_action_dist,
            policy_action = z,
        )

    def soft_update(self):
        pass

    def encode(self, states, keep_grad = False):
        return states

    def dist(self, batch, mode = "policy"): # 이제 이게 필요가 없음. 
        """
        """
        if mode == "consistency":
            states, G = batch.states, batch.relabeled_goals
            policy_skill = self.policy(torch.cat((states, G), dim = -1))
            policy_skill = torch.tanh(policy_skill)
            skill_consistency = F.mse_loss(policy_skill, batch.actions)

            return edict(
                skill_consistency = skill_consistency
            )

        else:
            states, G = batch.states, batch.G    
            policy_skill = self.policy(torch.cat((states, G), dim = -1))
            policy_skill = torch.tanh(policy_skill)

            return edict(
                policy_skill = policy_skill,
            )

    @torch.no_grad()
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
        dist_inputs = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        skill = self.dist(dist_inputs).policy_skill
        # TODO explore 여부에 따라 mu or sample을 결정
        return to_skill_embedding(skill)

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

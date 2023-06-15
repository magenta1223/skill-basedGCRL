import torch
from torch.nn import functional as F
from ...modules import BaseModule, ContextPolicyMixin
from ...utils import *
from ...contrib import TanhNormal
from easydict import EasyDict as edict
import d4rl


class SPiRL_Prior(ContextPolicyMixin, BaseModule):
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
        states = batch.states
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        prior = self.skill_prior.dist(states[:, 0])

        return edict(
            prior = prior,
            states = states,
            # policy_skill = policy_skill,
        )

    def encode(self, states, keep_grad = False):
        return states

        
    def dist(self, batch, mode = "policy"): # 이제 이게 필요가 없음. 
        """
        """
        states = batch.states
        
        with torch.no_grad():
            prior = self.skill_prior.dist(states)
            if self.tanh:
                prior_dist = prior._normal.base_dist
            else:
                prior_dist = prior.base_dist
            prior_locs, prior_scales = prior_dist.loc.clone().detach(), prior_dist.scale.clone().detach()
            prior_pre_scales = inverse_softplus(prior_scales)
            
        res_locs, res_pre_scales = self.highlevel_policy(states).chunk(2, dim=-1)

        # 혼합
        locs = res_locs + prior_locs
        scales = F.softplus(res_pre_scales + prior_pre_scales)
        policy_skill = get_dist(locs, scale = scales, tanh = self.tanh)

        return edict(
            prior = prior,
            policy_skill = policy_skill,
        )

    @torch.no_grad()
    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
        dist_inputs = edict(
            states = prep_state(states, self.device),
        )

        dist = self.dist(dist_inputs).policy_skill
        # TODO explore 여부에 따라 mu or sample을 결정
        if self.tanh:
            z_normal, z = dist.rsample_with_pre_tanh_value()
            return to_skill_embedding(z_normal), to_skill_embedding(z)

        else:
            return to_skill_embedding(z)

    def get_rl_params(self):
        

        return edict(
            policy = [
                {"params" : self.highlevel_policy.parameters(), "lr" : self.cfg.policy_lr}
            ],
            consistency = {
            }
        )

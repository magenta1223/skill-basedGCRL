import math
import numpy as np
import torch
import torch.nn as nn
from .base import SequentialBuilder, ContextPolicyMixin
from ..utils import *


class InverseDynamicsMLP(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None

    # for compatibility with simpl
    def dist_param(self, state):
        only_states = state[:, : self.in_feature]
        out = self(only_states)
        mu, log_std = out.chunk(2, dim = 1)
        return mu, log_std.clamp(-10, 2)

    def dist(self, state, subgoal, tanh = False):        
        id_inputs = torch.cat((state, subgoal), dim = -1)
        dist_params = self(id_inputs)
        dist, dist_detached = get_dist(dist_params, tanh= tanh), get_dist(dist_params.clone().detach(), tanh= tanh)
        
        return dist, dist_detached

class DecoderNetwork(ContextPolicyMixin, SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None
        self.log_sigma = nn.Parameter(-50*torch.ones(self.out_dim)) # for stochastic sampling
    #     self.visual_encoder = None

    # def set_visual_encoder(self, visual_encoder):
    #     if visual_encoder is not None:
    #         visual_encoder = deepcopy(visual_encoder)
    #         visual_encoder.eval()
    #         visual_encoder.requires_grad_(False)

    #     self.visual_encoder = visual_encoder 

    # from simpl. for compatibility with simpl
    def dist(self, batch_state_z):
        if self.state_dim is not None:
            # self.state_dim = 30
            if self.visual_encoder is not None:
                N = batch_state_z.shape
                visual_feature = self.visual_encoder(batch_state_z[:, 4:1028].view(N, 1, 32, 32))

                batch_state_z = torch.cat([
                    # batch_state_z[..., :self.state_dim],
                    visual_feature,
                    batch_state_z[..., -self.z_dim:]
                ], dim=-1)

            else:
                batch_state_z = torch.cat([
                    batch_state_z[..., :self.state_dim],
                    batch_state_z[..., -self.z_dim:]
                ], dim=-1)                

        loc = self(batch_state_z)
        log_scale = self.log_sigma[None, :].expand(len(loc), -1)
        dist = get_dist(loc, log_scale)
        return dist

    def act(self, state):
        if self.z is None:
            raise RuntimeError('z is not set')
        state = np.concatenate([state, self.z], axis=0)

        if self.explore is None:
            raise RuntimeError('explore is not set')
        
        batch_state = prep_state(state, self.device)


        with torch.no_grad():
            training = self.training
            self.eval()
            dist = self.dist(batch_state)
            self.train(training)

        if self.explore is True:
            batch_action = dist.sample()
        else:
            batch_action = dist.mean
        return batch_action.squeeze(0).cpu().numpy()
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, states):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        N, T = states.shape[:2]
        pos = (states[:, :, :2] * 100).to(int)
        return self.pe[ pos.view(-1) ].view(N, T, -1)

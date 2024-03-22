import math
from typing import Dict
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
        self.log_sigma = nn.Parameter(-50*torch.ones(self.out_dim)) 

    def dist(self, batch_state_z):
        if self.state_dim is not None:
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
    
class Learned_Distribution(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)

    def forward(self, x):
        dummy_input = torch.randn(x.shape[0], self.in_feature).to(self.device)
        return super().forward(dummy_input)

class Multisource_Encoder(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        del self.layers
        
        
        if config['env_state_dim'] == 0:
            ppc_config = {**config}
            ppc_config['in_feature'] = config['ppc_state_dim']
            ppc_config['out_dim'] = config['latent_state_dim']
            self.ppc_encoder = SequentialBuilder(ppc_config)
            self.env_encoder = None

        else:
            ppc_config = {**config}
            ppc_config['in_feature'] = config['ppc_state_dim']
            ppc_config['out_dim'] = config['latent_state_dim'] // 2
            self.ppc_encoder = SequentialBuilder(ppc_config)

            env_config = {**config}
            env_config['in_feature'] = config['env_state_dim']
            env_config['out_dim'] = config['latent_state_dim'] // 2
            self.env_encoder = SequentialBuilder(env_config)

    def forward(self, x):
        if self.env_state_dim == 0:
            ppc_embedding = self.ppc_encoder(x)
            return ppc_embedding, ppc_embedding, None
    
        else:
            ppc_state = x[..., :self.ppc_state_dim]
            env_state = x[..., self.ppc_state_dim :]

            ppc_embedding = self.ppc_encoder(ppc_state)
            env_embedding = self.env_encoder(env_state)
        
            return torch.cat((ppc_embedding, env_embedding), dim = -1), ppc_embedding, env_embedding

class Multisource_Decoder(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        del self.layers
    
        if config['env_state_dim'] == 0:
            ppc_config = {**config}
            ppc_config['in_feature'] = config['latent_state_dim']
            ppc_config['out_dim'] = config['ppc_state_dim']
            self.ppc_decoder = SequentialBuilder(ppc_config)

            self.env_decoder = None
        else:
            ppc_config = {**config}
            ppc_config['in_feature'] = config['latent_state_dim'] // 2
            ppc_config['out_dim'] = config['ppc_state_dim']
            self.ppc_decoder = SequentialBuilder(ppc_config)

            env_config = {**config}
            env_config['in_feature'] = config['latent_state_dim'] // 2
            env_config['out_dim'] = config['env_state_dim']
            self.env_decoder = SequentialBuilder(env_config)

    def forward(self, state_embedding):
        if self.env_state_dim == 0:
            ppc_state_hat = self.ppc_decoder(state_embedding)
            return ppc_state_hat, ppc_state_hat, None
        
        else:
            ppc_embedding = state_embedding[..., :self.latent_state_dim // 2]
            env_embedding = state_embedding[..., self.latent_state_dim // 2:]

            ppc_state_hat = self.ppc_decoder(ppc_embedding)
            env_state_hat = self.env_decoder(env_embedding)
        
            return torch.cat((ppc_state_hat, env_state_hat), dim = -1), ppc_state_hat,  env_state_hat


class SubgoalGenerator(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        self.embed_state = nn.Linear(self.latent_state_dim, self.hidden_dim)
        self.embed_goal = nn.Linear(self.n_goal, self.hidden_dim)
        # self.attn_blocks = nn.ModuleList([ TransformerBlock() for _ in range(3)])
        self.proj = nn.Linear(self.hidden_dim, self.latent_state_dim,  bias= False)

    def forward(self, state, goal):
        state_embedding = self.embed_state(state) # N, D
        goal_embedding = self.embed_goal(goal) # N, D
        for layer in self.layers:
            state_embedding = layer(state_embedding, goal_embedding, goal_embedding)

        return self.proj(state_embedding)

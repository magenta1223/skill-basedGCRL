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
        # 굳이 필요할까 ?
        if self.state_dim is not None:
            # self.state_dim = 30
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
    
class Normal_Distribution(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        del self.layers

    def dist(self, states):
        dummy_input = torch.zeros(states.shape[0], self.action_dim * 2)
        return get_fixed_dist(dummy_input, tanh = self.tanh)


class Multisource_Encoder(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        del self.layers
        
        
        # if config['env_state_dim'] == 0:
        #     ppc_config = {**config}
        #     ppc_config['in_feature'] = config['ppc_state_dim']
        #     ppc_config['out_dim'] = config['latent_state_dim']
        #     self.ppc_encoder = SequentialBuilder(ppc_config)
        #     self.env_encoder = None

        # else:
        #     ppc_config = {**config}
        #     ppc_config['in_feature'] = config['ppc_state_dim']
        #     ppc_config['out_dim'] = config['latent_state_dim'] // 2
        #     self.ppc_encoder = SequentialBuilder(ppc_config)

        #     env_config = {**config}
        #     env_config['in_feature'] = config['env_state_dim']
        #     env_config['out_dim'] = config['latent_state_dim'] // 2
        #     self.env_encoder = SequentialBuilder(env_config)

        pos_config = {**config}
        pos_config['in_feature'] = config['pos_state_dim']
        pos_config['out_dim'] = config['latent_state_dim'] // 2
        self.pos_encoder = SequentialBuilder(pos_config)

        nonPos_config = {**config}
        nonPos_config['in_feature'] = config['nonPos_state_dim']
        nonPos_config['out_dim'] = config['latent_state_dim'] // 2
        self.nonPos_encoder = SequentialBuilder(nonPos_config)



    def forward(self, x):
        # if self.env_state_dim != 0:
        #     ppc_state = x[..., :self.pos_state_dim]
        #     env_state = x[..., self.pos_state_dim :]

        #     ppc_embedding = self.ppc_encoder(ppc_state)
        #     env_embedding = self.env_encoder(env_state)
        
        #     return torch.cat((ppc_embedding, env_embedding), dim = -1), ppc_embedding, env_embedding
        
        # else:
        #     ppc_embedding = self.ppc_encoder(x)
        #     return ppc_embedding, ppc_embedding, None
    
        pos_state = x[..., :self.pos_state_dim]
        nonPos_state = x[..., self.pos_state_dim :]

        pos_embedding = self.pos_encoder(pos_state)
        nonPos_embedding = self.nonPos_encoder(nonPos_state)
    
        return torch.cat((pos_embedding, nonPos_embedding), dim = -1), pos_embedding, nonPos_embedding
        



class Multisource_Decoder(SequentialBuilder):
    def __init__(self, config: Dict[str, None]):
        super().__init__(config)
        del self.layers
    
        # if config['env_state_dim'] == 0:
        #     ppc_config = {**config}
        #     ppc_config['in_feature'] = config['latent_state_dim']
        #     ppc_config['out_dim'] = config['ppc_state_dim']
        #     self.ppc_decoder = SequentialBuilder(ppc_config)

        #     self.env_decoder = None
        # else:
        #     ppc_config = {**config}
        #     ppc_config['in_feature'] = config['latent_state_dim'] // 2
        #     ppc_config['out_dim'] = config['ppc_state_dim']
        #     self.ppc_decoder = SequentialBuilder(ppc_config)

        #     env_config = {**config}
        #     env_config['in_feature'] = config['latent_state_dim'] // 2
        #     env_config['out_dim'] = config['env_state_dim']
        #     self.env_decoder = SequentialBuilder(env_config)

        # if config['env_state_dim'] == 0:
        pos_config = {**config}
        pos_config['in_feature'] = config['latent_state_dim'] // 2
        pos_config['out_dim'] = config['pos_state_dim']
        self.pos_decoder = SequentialBuilder(pos_config)

        nonPos_config = {**config}
        nonPos_config['in_feature'] = config['latent_state_dim'] // 2
        nonPos_config['out_dim'] = config['nonPos_state_dim']
        self.nonPos_decoder = SequentialBuilder(nonPos_config)



    def forward(self, state_embedding):
        # if self.env_state_dim != 0:
        #     ppc_embedding = state_embedding[..., :self.latent_state_dim // 2]
        #     env_embedding = state_embedding[..., self.latent_state_dim // 2:]

        #     ppc_state_hat = self.ppc_decoder(ppc_embedding)
        #     env_state_hat = self.env_decoder(env_embedding)
        
        #     return torch.cat((ppc_state_hat, env_state_hat), dim = -1), ppc_state_hat,  env_state_hat
        # else:
        #     ppc_state_hat = self.ppc_decoder(state_embedding)
        #     return ppc_state_hat, ppc_state_hat, None

        pos_embedding = state_embedding[..., :self.latent_state_dim // 2]
        nonPos_embedding = state_embedding[..., self.latent_state_dim // 2:]

        pos_state_hat = self.pos_decoder(pos_embedding)
        nonPos_state_hat = self.nonPos_decoder(nonPos_embedding)
    
        return torch.cat((pos_state_hat, nonPos_state_hat), dim = -1), pos_state_hat, nonPos_state_hat


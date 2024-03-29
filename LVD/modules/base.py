from contextlib import contextmanager
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_dist


class BaseModule(nn.Module):
    """
    Module class equipped with common methods for logging & distributions 
    """
    def __init__(self, cfg):
        super(BaseModule, self).__init__()
        self.set_attrs(cfg)
        self._device = nn.Parameter(torch.zeros(1))

    # set configs
    def set_attrs(self, cfg = None):
        if cfg is not None:
            for k, v in cfg.items():
                setattr(self, k, deepcopy(v))           

    def get(self, name):
        if not name : #name is None or not name:
            return None
        return getattr(self, name)

    def forward(self, x):
        return NotImplementedError

    @property
    def device(self):
        return self._device.device


class SequentialBuilder(BaseModule):
    def __init__(self, config : Dict[str, None]):
        super().__init__(config)
        self.build()
        self.explore = None

    def get(self, name):
        if not name : #name is None or not name:
            return None
        return getattr(self, name)

    def layerbuild(self, attr_list, repeat = None):
        build =  [[ self.get(attr_nm) for attr_nm in attr_list  ]]
        if repeat is not None:
            build = build * repeat
        return build 

    def get_build(self):
        if self.module_type == "rnn":
            build = self.layerbuild(["linear_cls", "in_feature", "hidden_dim", None, "act_cls", "bias"])
            # build += self.layerbuild(["rnn_cls", "hidden_dim", "hidden_dim", "n_blocks", "bias", "batch_first", "dropout"])
            build += self.layerbuild(["rnn_cls", "hidden_dim", "hidden_dim", "n_blocks", "bias", "batch_first", "dropout"], self.get("n_blocks"))
            build += self.layerbuild(["linear_cls", "hidden_dim", "out_dim", None, None, "proj_bias"])


        elif self.module_type == "linear":
            build = self.layerbuild(["linear_cls", "in_feature", "hidden_dim", None, "act_cls", "bias", "dropout"])
            build += self.layerbuild(["linear_cls", "hidden_dim", "hidden_dim", "norm_cls", "act_cls"], self.get("n_blocks"))
            build += self.layerbuild(["linear_cls", "hidden_dim", "out_dim", None, None,  "bias", "dropout"])


        elif self.module_type == "transformer":
            build = self.layerbuild(["block_cls", "hidden_dim", "act_cls"], self.get("n_blocks")) 

        else:
            build = NotImplementedError

        return build

    def build(self):
        build = self.get_build()
        layers = []
        for args in build:
            cls, args = args[0], args[1:]
            layers.append(cls(*args))
        self.layers = nn.ModuleList(layers)


    def forward(self, x, *args, **kwargs):
        out = x 
        for layer in self.layers:
            out = layer(out)
            if isinstance(out, tuple): # rnn
                out = out[0]
        return out


    def dist(self, *args, detached = False):
        result = self(*args)

        if self.module_type in ['rnn', 'transformer']:
            if self.return_last:
                result = result[:, -1]

        if detached:
            return get_dist(result, tanh = self.tanh), get_dist(result, detached= True, tanh = self.tanh)
        else:
            return get_dist(result, tanh = self.tanh)



class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_cls = None, act_cls = None, bias = False, dropout = 0):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim,  bias = bias)]
        if norm_cls is not None:
            layers.append(norm_cls(out_dim))

        if act_cls is not None:
            if act_cls == nn.LeakyReLU:
                layers.append(act_cls(0.2, True))
            else:
                layers.append(act_cls(inplace= True))
        
        if dropout != 0:
            layers.append(nn.Dropout1d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        # hidden dim만 있으면 됨. 
        super(SimpleAttention, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        q, k, v = self.q(query), self.k(key), self.v(value)
        q, k, v = q.unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

        attn = q.matmul(k.permute(0,2,1)) 
        attn_score = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn_score, v)
        return  attn_out

class FFN(nn.Module):
    def __init__(self, hidden_dim, act_cls):
        # hiddden_dim 
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = act_cls()
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, act_cls) -> None:
        super(TransformerBlock, self).__init__()

        self.attn = SimpleAttention(hidden_dim)
        self.ffn = FFN(hidden_dim, act_cls)

    def forward(self, query, key, value):
        attn_out = self.attn(query, key, value).squeeze(-1)
        ffn_out = self.ffn(attn_out)
        return ffn_out

# from simpl
class ContextPolicyMixin:
    z_dim = NotImplemented
    z = None
    explore = False

    @contextmanager
    def condition(self, z):
        if type(z) != np.ndarray or z.shape != (self.z_dim, ):
            raise ValueError(f'z should be np.array with shape {self.z_dim}, but given : {z}')
        prev_z = self.z
        self.z = z
        yield
        self.z = prev_z

    def act(self, state):
        if self.z is None:
            raise RuntimeError('z is not set')
        state_z = np.concatenate([state, self.z], axis=0)
        return super(ContextPolicyMixin, self).act(state_z)

    def dist(self, batch_state_z, tanh = False):
        return super(ContextPolicyMixin, self).dist(batch_state_z, tanh= tanh)

    def dist_with_z(self, batch_state, batch_z, tanh = False):
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.dist(batch_state_z, tanh= tanh)
    
    @staticmethod
    def transform_numpy(x):
        return x.detach().cpu().squeeze(0).numpy()

    # from simpl
    @contextmanager
    def no_expl(self):
        explore = self.explore
        self.explore = False
        yield
        self.explore = explore

    @contextmanager
    def expl(self):
        explore = self.explore
        self.explore = True
        yield
        self.explore = explore



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        
        return x.view(-1, np.prod(x.shape[1:]))
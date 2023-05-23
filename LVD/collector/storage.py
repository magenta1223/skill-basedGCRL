from collections import deque, OrderedDict
from copy import deepcopy

from ..contrib.simpl.collector.storage import  Batch, Buffer, Episode
import numpy as np
import torch
from torch.nn import functional as F
from ..utils import prep_state
from easydict import EasyDict as edict


class Episode_G(Episode):
    def __init__(self, init_state):
        super().__init__(init_state)
        self.__G__ = []

    def __repr__(self):
        return f'Episode(cum_reward={sum(self.rewards)}, length={len(self)})'

    def __len__(self):
        return len(self.actions)
    
    def add_step(self, action, next_state, G, reward, done, info):
        super().add_step( action, next_state, reward, done, info)
        self.__G__.append(G)


    def __repr__(self):
        return f'Episode(cum_reward={sum(self.rewards)}, length={len(self)}, G = )'
    
    def as_transitions(self):
        return torch.as_tensor(np.concatenate([
            self.states,
            self.actions,
            self.next_states,
            self.G,
            self.rewards[:, None],
            self.dones[:, None],
        ], axis=-1))


    @property
    def G(self):
        return np.array(self.__G__)


class Batch_G(Batch):
    def __init__(self, states, actions, next_states, G, rewards, dones, transitions=None):
        # super().__init__(states, actions, rewards, dones, next_states, transitions)

        self.data = OrderedDict([
            ('states', states),
            ('actions', actions),
            ('next_states', next_states),
            ('G', G),
            ('rewards', rewards),
            ('dones', dones),
        ])
        self.transitions = transitions


    def parse(self, tanh = False):
        batch_dict = edict()

        batch_dict['states'] = prep_state(self.states, self.device) #prep_state(batch.states, self.device)
        batch_dict['next_states'] = prep_state(self.next_states, self.device)
        batch_dict['G'] = prep_state(self.G, self.device) 
        batch_dict['rewards'] = self.rewards
        batch_dict['dones'] = self.dones

        actions = prep_state(self.actions, self.device) # high-actions

        if tanh:
            batch_dict['actions'], batch_dict['actions_normal'] = actions.chunk(2, -1)
        else:
            batch_dict['actions'] = actions

        return batch_dict
    
    @property
    def G(self):
        return self.data['G']


class Buffer_G(Buffer):
    """
    Override 
    H-step state를 다.. 얻어놔야 함. 
    enqueue해서 다 얻어놨고, 이게.. H-step을 쓸 수 있는게 있고 아닌게 있음. 
    그냥 따로 구성하는게 .. 
    """
    def __init__(self, state_dim, action_dim, goal_dim, max_size, tanh = False):
        super().__init__(state_dim, action_dim, max_size)
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.max_size = max_size
        
        # self.ptr = 0
        # self.size = 0
        
        # self.episodes = deque()
        # self.episode_ptrs = deque()
        
        # for tanh
        self.tanh = tanh
        action_dim *= 2

        # states, next_states, skill, rewards, dones, 
        self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 2)
        
        dims = OrderedDict([
            ('state', state_dim),
            ('action', action_dim),
            ('next_state', state_dim),
            ('G', goal_dim),
            ('reward', 1),
            ('done', 1),
        ])
        self.layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.layout[k] = slice(prev_i, next_i)
            prev_i = next_i
        
        self.device = None

    @property
    def G(self):
        return self.transitions[:, self.layout['G']]

    def sample(self, n):
        indices = torch.randint(self.size, size=[n], device=self.device)
        transitions = self.transitions[indices]
        return Batch_G(*[transitions[:, i] for i in self.layout.values()], transitions).to(self.device).parse(self.tanh)


class Buffer_TT(Buffer):
    """
    Buffer for through time
    N, T, D 형태로 구성. 연속된 skill transition이 나오도록. 
    """

    def __init__(self, state_dim, action_dim, n_skill, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_skill = n_skill
        self.max_size = max_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        self.transitions = torch.empty(max_size, n_skill, 2*state_dim + action_dim + 2)
        
        dims = OrderedDict([
            ('state', state_dim),
            ('action', action_dim),
            ('reward', 1),
            ('done', 1),
            ('next_state', state_dim),
        ])
        self.layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.layout[k] = slice(prev_i, next_i)
            prev_i = next_i
        
        self.device = None
  
    def __repr__(self):
        return f'Buffer(max_size={self.max_size}, self.size={self.size})'

    def to(self, device):
        self.device = device
        return self

    @property
    def states(self):
        return self.transitions[:, :, self.layout['state']]
    
    @property
    def actions(self):
        return self.transitions[:, :,  self.layout['action']]
    
    @property
    def rewards(self):
        return self.transitions[:, :, self.layout['reward']]

    @property
    def dones(self):
        return self.transitions[:, :, self.layout['done']]

    @property
    def next_states(self):
        return self.transitions[:, :, self.layout['next_state']]

    def enqueue(self, episode):
        # 에피소드 길이가 0 이 될 때 까지
        while len(self.episodes) > 0:
            # 가장 앞의 에피소드와
            old_episode = self.episodes[0]
            # 그 ptr을 가져옴
            ptr = self.episode_ptrs[0] 
            # ptr - self.ptr을 최대크기 (20000) 으로 나눈 몫임. 
            dist = (ptr - self.ptr) % self.max_size

            # 
            if dist < len(episode):
                self.episodes.popleft()
                self.episode_ptrs.popleft()
            else:
                break

        # self.ptr을 더한 후 
        self.episodes.append(episode)
        self.episode_ptrs.append(self.ptr)

        # transition으로 만듬
        transitions = torch.as_tensor(np.concatenate([
            episode.states[:-1], episode.actions,
            np.array(episode.rewards)[:, None], np.array(episode.dones)[:, None],
            episode.states[1:]
        ], axis=-1)).unfold(dimension= 0, size = self.n_skill, step = 1).permute(0, 2, 1)

        if self.ptr + len(transitions) - 1 <= self.max_size:
            # 빈 곳에 할당
            self.transitions[self.ptr:self.ptr+len(transitions)] = transitions
        # 만약 1배 이상 2배 이하라면? 
        elif self.ptr + len(transitions) -1 < 2*self.max_size:
            # 잘라서 앞에넣고 뒤에넣고
            self.transitions[self.ptr:] = transitions[:self.max_size-self.ptr]
            self.transitions[:len(transitions)-self.max_size+self.ptr] = transitions[self.max_size-self.ptr:]
        else:
            raise NotImplementedError

        # 즉, ptr은 현재 episode를 더하고 난 후의 위치임. 
        self.ptr = (self.ptr + len(transitions) - 1) % self.max_size
        self.size = min(self.size + len(transitions)-1 , self.max_size)


    def sample(self, n):
        indices = torch.randint(self.size, size=[n], device=self.device)
        transitions = self.transitions[indices] # N, n_skill, D 
        return Batch(*[transitions[:, :, i] for i in self.layout.values()], transitions)




class Offline_Buffer:
    def __init__(self, state_dim, action_dim, trajectory_length = 10, max_size = 1000) -> None:

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.max_size = max_size

        self.size = 0 
        self.pos = 0  

        self.states = torch.empty(max_size, trajectory_length + 1, state_dim)
        self.actions = torch.empty(max_size, trajectory_length, action_dim)

    def enqueue(self, states, actions):
        N, T, _ = actions.shape
        self.size = min(  self.size + N, self.max_size)        
        # if exceed max size
        if self.max_size < self.pos + N:
            self.states[self.pos : self.max_size] = states[: self.max_size - self.pos]
            self.actions[self.pos : self.max_size] = actions[: self.max_size - self.pos]
            self.pos = 0
            # remainder 
            states = states[self.max_size - self.pos : ]
            actions = actions[self.max_size - self.pos : ]

        N = states.shape[0]
        self.states[self.pos : self.pos + N] = states
        self.actions[self.pos : self.pos + N] = actions

        self.pos += N


    def sample(self):
        i = np.random.randint(0, self.size)

        states = self.states[i].numpy()
        actions = self.actions[i].numpy()

        return states, actions


    def copy_from(self, buffer):
        self.states = buffer.states.clone()
        self.actions = buffer.actions.clone()
        self.size = buffer.size
        self.pos = buffer.pos
        print(f"Buffer Size : {self.size}")
        
    def reset(self):
        
        self.size = 0 # 현재 buffer에 차 있는 subtrajectories의 전체 길이
        self.pos = 0  # 다음에 어디에 추가할지. 

        self.states = torch.empty(self.max_size, self.trajectory_length + 1, self.state_dim)
        self.actions = torch.empty(self.max_size, self.trajectory_length, self.action_dim)
        print(  "Buffer Reset. Size : ", self.size)
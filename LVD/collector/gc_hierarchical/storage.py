from collections import deque, OrderedDict
from copy import deepcopy

from ...contrib.simpl.collector.storage import  Batch, Buffer, Episode
import numpy as np
import torch
from torch.nn import functional as F
from ...utils import prep_state, StateProcessor
from easydict import EasyDict as edict
import random

class GC_Batch(Batch):
    def __init__(self, states, actions, rewards, relabeled_rewards, dones, next_states, goals, relabeled_goals, transitions=None, tanh = False):
        # def   __init__(states, actions, rewards, dones, next_states, transitions=None):
        super().__init__(states, actions, rewards, dones, next_states, transitions)

        self.data['goals'] = goals
        self.data['relabeled_goals'] = relabeled_goals
        self.data['relabeled_rewards'] = relabeled_rewards

        self.tanh = tanh

    @property
    def goals(self):
        return self.data['goals']
    
    @property
    def relabeled_goals(self):
        return self.data['relabeled_goals']

    @property
    def relabeled_rewards(self):
        return self.data['relabeled_rewards']

    def parse(self):
        
        relabel = random.random() > 0.8 

        

        # relabel = True
        batch_dict = edict(
            states = self.states,
            next_states = self.next_states,
            rewards = self.relabeled_rewards if relabel else self.rewards ,
            dones = self.dones,
            G = self.relabeled_goals if relabel else self.goals
        )
        if self.tanh:
            batch_dict['actions'], batch_dict['actions_normal'] = self.actions.chunk(2, -1)
        else:
            batch_dict['actions'] = self.actions

        return batch_dict

class GC_Buffer(Buffer):
    """
    Override 
    H-step state를 다.. 얻어놔야 함. 
    enqueue해서 다 얻어놨고, 이게.. H-step을 쓸 수 있는게 있고 아닌게 있음. 
    그냥 따로 구성하는게 .. 
    """
    def __init__(self, state_dim, action_dim, goal_dim, max_size, env_name, tanh = False):

        self.state_processor = StateProcessor(env_name)


        self.tanh = tanh
        if self.tanh:
            action_dim *= 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones


        self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim * 2 + 3) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', state_dim),
            ('action', action_dim),
            ('reward', 1),
            ('relabeled_reward', 1),
            ('done', 1),
            ('next_state', state_dim),
            ('goal', goal_dim),
            ('relabled_goal', goal_dim),
        ])
        self.layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.layout[k] = slice(prev_i, next_i)
            prev_i = next_i
        
        self.device = None



    @property
    def goal(self):
        return self.transitions[:, self.layout['goal']]

    @property
    def relabled_goal(self):
        return self.transitions[:, self.layout['relabled_goal']]

    @property
    def relabeled_rewards(self):
        return self.transitions[:, self.layout['relabeled_rewards']]




    def ep_to_transtions(self, episode):
        len_ep = len(episode.states[:-1])

        # last state = goal로 변경
        # relabeled reward 추가
        relabeled_goal = self.state_processor.goal_transform(np.array(episode.states[-1]))
        relabeled_goals = np.tile(relabeled_goal, (len(episode.states)-1, 1))

        return torch.as_tensor(np.concatenate([
            episode.states[:-1],
            episode.actions,
            np.array(episode.rewards)[:, None],
            np.array(episode.relabeled_rewards)[:, None],
            np.array(episode.dones)[:, None],
            episode.states[1:],
            episode.goals,
            relabeled_goals
        ], axis=-1))

    def sample(self, n):
        indices = torch.randint(self.size, size=[n])
        transitions = self.transitions[indices]
        batch = GC_Batch(*[transitions[:, i] for i in self.layout.values()], transitions, self.tanh).to(self.device)
        return batch.parse()


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
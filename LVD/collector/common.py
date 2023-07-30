from collections import deque, OrderedDict
from copy import deepcopy
from easydict import EasyDict as edict


import random
import numpy as np
import torch
from ..utils import StateProcessor
from ..contrib.simpl.collector.storage import  Batch, Buffer, Episode


class Episode_RR(Episode):
    """
    Episode for relabeled reward 
    """
    def __init__(self, init_state, env_name = None):
        super().__init__(init_state)
        self.relabeled_rewards = []
        self.goals = []
        if env_name is not None:
            self.state_processor = StateProcessor(env_name)

    def add_step(self, action, next_state, goal, reward, relabeled_reward, done, info):
        super().add_step(action, next_state, reward, done, info)
        self.relabeled_rewards.append(relabeled_reward)
        self.goals.append(goal)


class HierarchicalEpisode_RR(Episode_RR):
    def __init__(self, init_state):
        super().__init__(init_state)
        # self.low_actions = self.actions
        self.high_actions = []
    
    def add_step(self, low_action, high_action, next_state, goal, reward, relabeled_reward, done, info):
        # MDP transitions
        super().add_step(low_action, next_state, goal, reward, relabeled_reward, done, info)
        self.high_actions.append(high_action)

    def as_high_episode(self):
        """
        high-action은 H-step마다 값이 None 이 아
        """

        high_episode = Episode_RR(self.states[0])

        prev_t = 0
        for t in range(1, len(self)):
            if self.high_actions[t] is not None:
                # high-action은 H-step마다 값이 None이 아니다.
                # raw episode를 H-step 단위로 끊고, action을 high-action으로 대체해서 넣음. 
                high_episode.add_step(
                    self.high_actions[prev_t],
                    self.states[t],
                    self.goals[t],
                    sum(self.rewards[prev_t:t]),
                    sum(self.relabeled_rewards[prev_t:t]),
                    self.dones[t],
                    self.infos[t]
                )

                prev_t = t
        
        high_episode.add_step(
            self.high_actions[prev_t],
            self.states[-1],
            self.goals[t],
            sum(self.rewards[prev_t:]), 
            sum(self.relabeled_rewards[prev_t:]),
            self.dones[-1], 
            self.infos[-1]
        )

        high_episode.raw_episode = self

        return high_episode 




class HierarchicalEpisode_Relabel(Episode_RR):
    def __init__(self, init_state, env_name = None):
        super().__init__(init_state, env_name)
        # self.low_actions = self.actions
        self.high_actions = []
    
    def add_step(self, low_action, high_action, next_state, goal, reward, relabeled_reward, done, info):
        # MDP transitions
        super().add_step(low_action, next_state, goal, reward, relabeled_reward, done, info)
        self.high_actions.append(high_action)

    def as_high_episode(self):
        """
        high-action은 H-step마다 값이 None 이 아
        """

        high_episode = Episode_RR(self.states[0])
        prev_t = 0

        for t in range(1, len(self)):
            if self.high_actions[t] is not None:
                # high-action은 H-step마다 값이 None이 아니다.
                # raw episode를 H-step 단위로 끊고, action을 high-action으로 대체해서 넣음. 
                high_episode.add_step(
                    self.high_actions[prev_t],
                    self.states[t],
                    self.goals[t],
                    sum(self.rewards[prev_t:t]),
                    sum(self.relabeled_rewards[prev_t:t]),
                    self.dones[t],
                    self.infos[t]
                )

                prev_t = t
        
        # 남은거 
        high_episode.add_step(
            self.high_actions[prev_t],
            self.states[-1],
            self.goals[t],
            sum(self.rewards[prev_t:]), 
            sum(self.relabeled_rewards[prev_t:]),
            self.dones[-1], 
            self.infos[-1]
        )
        high_episode.raw_episode = self

        # ----------------------------------------------------------------------------------------------- #


        prev_t = 0
        relabeled_rewards = [ i  for i, r in enumerate(self.relabeled_rewards) if r != 0]
        if len(relabeled_rewards):
            maxlen_relabeled = max([ i  for i, r in enumerate(self.relabeled_rewards) if r != 0])
        else:
            maxlen_relabeled = None
        
        if maxlen_relabeled is not None:
            high_episode_relabel = Episode_RR(self.states[0])
            relabeled_goal = self.states[maxlen_relabeled]
            relabeled_goal = self.state_processor.goal_transform(np.array(relabeled_goal))
            relabeled_dones = [ False for _ in range(maxlen_relabeled)]
            relabeled_dones[-1] = True

            for t in range(1, maxlen_relabeled):
                if self.high_actions[t] is not None:
                    high_episode_relabel.add_step(
                        self.high_actions[prev_t],
                        self.states[t],
                        relabeled_goal, # relabeled goal로 대체 
                        sum(self.relabeled_rewards[prev_t:t]), 
                        sum(self.rewards[prev_t:t]), # 마찬가지 
                        relabeled_dones[t],
                        self.infos[t]
                    )

                    prev_t = t

            high_episode_relabel.add_step(
                self.high_actions[prev_t],
                self.states[-1],
                relabeled_goal,
                sum(self.relabeled_rewards[prev_t:]), 
                sum(self.rewards[prev_t:]), # 마찬가지 
                relabeled_dones[-1], 
                self.infos[-1]
            )
            high_episode_relabel.raw_episode = self

        else:
            high_episode_relabel = None


        return high_episode, high_episode_relabel


class GC_Batch(Batch):
    def __init__(self, states, actions, rewards, relabeled_rewards, dones, next_states, goals, relabeled_goals, transitions=None, tanh = False, hindsight_relabel = False):
        # def   __init__(states, actions, rewards, dones, next_states, transitions=None):
        super().__init__(states, actions, rewards, dones, next_states, transitions)

        self.data['goals'] = goals
        self.data['relabeled_goals'] = relabeled_goals
        self.data['relabeled_rewards'] = relabeled_rewards

        self.tanh = tanh
        self.hindsight_relabel = hindsight_relabel

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
        
        if self.hindsight_relabel:
            # sample별로 해야 함. 
            # indices = torch.randn(len(self.states), 1).cuda()
            # indices = torch.ones(len(self.states), 1).cuda() # relabeled 100% 
            indices = torch.zeros(len(self.states), 1).cuda() # relabeled 0% 

        else:
            indices = torch.zeros(len(self.states), 1).cuda()

        batch_dict = edict(
            states = self.states,
            next_states = self.next_states,

            # rewards = self.rewards,
            # G = self.goals,

            # rewards = self.relabeled_rewards if relabel else self.rewards,
            # G = self.relabeled_goals if relabel else self.goals,
            
            # GCQ
            # rewards = torch.where( indices < 1- 0.2, self.rewards, self.relabeled_rewards),
            # G = torch.where( indices < 1- 0.2, self.goals, self.relabeled_goals),
            rewards = self.rewards,
            G = self.goals,
            relabeled_G = self.relabeled_goals,
            dones = self.dones,
        )
        if self.tanh:
            batch_dict['actions'], batch_dict['actions_normal'] = self.actions.chunk(2, -1)
        else:
            batch_dict['actions'] = self.actions

        return batch_dict

class GC_Buffer(Buffer):
    """
    """
    # def __init__(self, state_dim, action_dim, goal_dim, max_size, env_name, tanh = False, hindsight_relabeling = False):
    def __init__(self, cfg):

        self.state_processor = StateProcessor(cfg.env_name)
        self.env_name = cfg.env_name


        self.tanh = cfg.tanh
        if self.tanh:
            action_dim *= 2
        self.hindsight_relabel = cfg.hindsight_relabel

        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.max_size = cfg.buffer_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones


        self.transitions = torch.empty(cfg.buffer_size, 2*cfg.state_dim + cfg.action_dim + cfg.n_goal * 2 + 3) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', cfg.action_dim),
            ('reward', 1),
            ('relabeled_reward', 1),
            ('done', 1),
            ('next_state', cfg.state_dim),
            ('goal', cfg.n_goal),
            ('relabled_goal', cfg.n_goal),
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

        relabeled_rewards = deepcopy(episode.relabeled_rewards)

        if self.env_name == "maze":
            relabeled_rewards[-1] = 100

        relabeled_rewards = np.array(relabeled_rewards)[:, None]


        return torch.as_tensor(np.concatenate([
            episode.states[:-1],
            episode.actions,
            np.array(episode.rewards)[:, None],
            relabeled_rewards,
            np.array(episode.dones)[:, None],
            episode.states[1:],
            episode.goals,
            relabeled_goals
        ], axis=-1))

    def sample(self, n):
        indices = torch.randint(self.size, size=[n])
        transitions = self.transitions[indices]
        batch = GC_Batch(*[transitions[:, i] for i in self.layout.values()], transitions, self.tanh, self.hindsight_relabel).to(self.device)
        return batch.parse()
    


class GC_Temporal_Buffer(Buffer):
    """
    """
    # def __init__(self, state_dim, action_dim, goal_dim, max_size, env_name, tanh = False, hindsight_relabeling = False):
    def __init__(self, cfg):
        
        self.cfg = cfg 
        self.state_processor = StateProcessor(cfg.env_name)
        self.env_name = cfg.env_name


        self.tanh = cfg.tanh
        if self.tanh:
            cfg.skill_dim *= 2
        self.hindsight_relabel = cfg.hindsight_relabel

        self.state_dim = cfg.state_dim
        self.action_dim = cfg.skill_dim
        self.max_size = cfg.buffer_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones


        self.transitions = torch.empty(cfg.buffer_size, cfg.n_skill, 2*cfg.state_dim + cfg.skill_dim + cfg.n_goal * 2 + 3) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', cfg.skill_dim),
            ('reward', 1),
            ('relabeled_reward', 1),
            ('done', 1),
            ('next_state', cfg.state_dim),
            ('goal', cfg.n_goal),
            ('relabled_goal', cfg.n_goal),
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

        relabeled_rewards = deepcopy(episode.relabeled_rewards)

        if self.env_name == "maze":
            relabeled_rewards[-1] = 100

        relabeled_rewards = np.array(relabeled_rewards)[:, None]

        transitions = []

        for i in range(len(episode.states)-self.cfg.n_skill - 1):
            _range = slice(i, i+self.cfg.n_skill)
            _next_range = slice(i + 1, i + 1 +self.cfg.n_skill)
            transition = np.concatenate([
                episode.states[_range],
                episode.actions[_range],
                np.array(episode.rewards[_range])[:, None],
                relabeled_rewards[_range],
                np.array(episode.dones[_range])[:, None],
                episode.states[_next_range],
                episode.goals[_range],
                relabeled_goals[_range]
            ], axis=-1)

            transitions.append(torch.as_tensor(transition))
        return torch.stack(transitions, dim = 0)
    
    def enqueue(self, episode):
        # 에피소드 길이가 0 이 될 때 까지
        # while len(self.episodes) > 0:
        #     # 가장 앞의 에피소드와
        #     old_episode = self.episodes[0]
        #     # 그 ptr을 가져옴
        #     ptr = self.episode_ptrs[0]
        #     # ptr - self.ptr을 최대크기 (20000) 으로 나눈 몫임. 
        #     dist = (ptr - self.ptr) % self.max_size

        #     # 
        #     if dist < len(episode):
        #         self.episodes.popleft()
        #         self.episode_ptrs.popleft()
        #     else:
        #         break

        # # self.ptr을 더한 후 
        # self.episodes.append(episode)
        # self.episode_ptrs.append(self.ptr)
        
        # transition으로 만듬
        transitions = self.ep_to_transtions(episode)

        
        # self.ptr + 에피소드 길이가 최대 크기 이하
        # 즉, 처음부터 채우고 있는 과정임.
        n_transition_toAdd = len(episode) - self.cfg.n_skill
        if self.ptr + n_transition_toAdd <= self.max_size:
            # 빈 곳에 할당
            self.transitions[self.ptr:self.ptr+n_transition_toAdd] = transitions
        # 만약 1배 이상 2배 이하라면? 
        elif self.ptr + n_transition_toAdd < 2*self.max_size:
            # 잘라서 앞에넣고 뒤에넣고
            self.transitions[self.ptr:] = transitions[:self.max_size-self.ptr]
            self.transitions[:n_transition_toAdd-self.max_size+self.ptr] = transitions[self.max_size-self.ptr:]
        else:
            raise NotImplementedError

        # 즉, ptr은 현재 episode를 더하고 난 후의 위치임. 
        self.ptr = (self.ptr + n_transition_toAdd) % self.max_size
        self.size = min(self.size + n_transition_toAdd, self.max_size)


    def sample(self, n):
        indices = torch.randint(self.size, size=[n])
        transitions = self.transitions[indices]
        batch = GC_Batch(*[transitions[..., i] for i in self.layout.values()], transitions, self.tanh, self.hindsight_relabel).to(self.device)
        return batch.parse()


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
        self.cs = torch.empty(max_size)

    def enqueue(self, states, actions, c = None):
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

            if c is not None:
                self.cs[self.pos : self.max_size] = c # c 는 scalar임. 

        N = states.shape[0]
        self.states[self.pos : self.pos + N] = states
        self.actions[self.pos : self.pos + N] = actions
        if c is not None:
            self.cs[self.pos : self.pos + N] = c

        self.pos += N


    def sample(self):
        i = np.random.randint(0, self.size)

        states = self.states[i].numpy()
        actions = self.actions[i].numpy()
        c = self.cs[i].numpy()

        return states, actions, c


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
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

class HierarchicalEpisode_Relabel(Episode_RR):
    def __init__(self, init_state, env_name = None, binary_reward = False):
        super().__init__(init_state, env_name)
        # self.low_actions = self.actions
        self.high_actions = []
        self.binary_reward = binary_reward
    
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

        # reward를 받을만한 것이 있었다면 ~ -> 어케암 
        # relabeled_rewards = [ i  for i, r in enumerate(self.relabeled_rewards) if r != 0]
        # if len(relabeled_rewards):
        #     maxlen_relabeled = max([ i  for i, r in enumerate(self.relabeled_rewards) if r != 0])
        # else:
        #     maxlen_relabeled = None
        
        # 마지막을 1로
        if self.binary_reward:
            # initialize episode
            high_episode_relabel = Episode_RR(self.states[0])
            prev_t = 0
        
            # goal relabeling ~ 을 이렇게 해야 할까? 
            # transition에서 하는게 나음. 
            relabeled_goal = self.state_processor.goal_transform(np.array(self.states[-1]))
            
            self.relabeled_rewards[-1] = 1

            for t in range(1, len(self)):
                if self.high_actions[t] is not None:
                    high_episode_relabel.add_step(
                        self.high_actions[prev_t],
                        self.states[t],
                        relabeled_goal, # relabeled goal로 대체 
                        sum(self.relabeled_rewards[prev_t:t]), 
                        sum(self.rewards[prev_t:t]), # 마찬가지 
                        self.dones[t],
                        self.infos[t]
                    )

                    prev_t = t

            high_episode_relabel.add_step(
                self.high_actions[prev_t],
                self.states[-1],
                relabeled_goal,
                sum(self.relabeled_rewards[prev_t:]), 
                sum(self.rewards[prev_t:]), # 마찬가지 
                self.dones[-1], 
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
        

        if "flat" in cfg.structure:
            action_dim = cfg.action_dim
        else:
            action_dim = cfg.skill_dim

        self.tanh = cfg.tanh
        if self.tanh:
            action_dim *= 2
        self.hindsight_relabel = cfg.hindsight_relabel

        self.state_dim = cfg.state_dim
        self.action_dim = action_dim
        self.max_size = cfg.buffer_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones


        self.transitions = torch.empty(cfg.buffer_size, 2*cfg.state_dim + action_dim + cfg.n_goal * 2 + 3) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', action_dim),
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
        
        # relabeling을 위해 
        # self.episodes가 있음. ep_index만 저장하면 된다. 
        # popleft 할 때 마다 -> 전체 ep_index 하나씩 빼주면 됨. 
        # 바뀜.

        # transition 만들 때 ep index 하나 만들고
        # ep 하나 추가할 때 마다 
        # 0에서 시작 -> popleft 안하면 +1 
        # 


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
            skill_dim = cfg.skill_dim * 2
        else:
            skill_dim = cfg.skill_dim
        self.hindsight_relabel = cfg.hindsight_relabel

        self.state_dim = cfg.state_dim
        self.action_dim = skill_dim
        self.max_size = cfg.buffer_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones

        self.transitions = torch.empty(cfg.buffer_size, cfg.n_skill, 2*cfg.state_dim + self.action_dim + cfg.n_goal * 2 + 3) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', self.action_dim),
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

class GC_Buffer_Relabel(Buffer):
    """
    """
    # def __init__(self, state_dim, action_dim, goal_dim, max_size, env_name, tanh = False, hindsight_relabeling = False):
    def __init__(self, cfg):
        self.cfg = cfg

        self.state_processor = StateProcessor(cfg.env_name)
        self.env_name = cfg.env_name
        

        if "flat" in cfg.structure:
            action_dim = cfg.action_dim
        else:
            action_dim = cfg.skill_dim

        self.tanh = cfg.tanh
        if self.tanh:
            action_dim *= 2
        self.hindsight_relabel = cfg.hindsight_relabel

        self.state_dim = cfg.state_dim
        self.action_dim = action_dim
        self.max_size = cfg.buffer_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        # self.transitions = torch.empty(max_size, 2*state_dim + action_dim + goal_dim + 3) # rwd, relabeled rwd, dones


        self.transitions = torch.empty(cfg.buffer_size, 2*cfg.state_dim + action_dim + cfg.n_goal + 4) # rwd, relabeled rwd, dones
        # states, next_states, action, goal, relabeled_goal, rwd, relabeled_rwd, dones 


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', action_dim),
            ('reward', 1),
            ('done', 1),
            ('next_state', cfg.state_dim),
            ('goal', cfg.n_goal),
            ('state_index', 1), # state index for relabeling 
            ('ep_index', 1), # episode index for relabeling 
        ])
        self.layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.layout[k] = slice(prev_i, next_i)
            prev_i = next_i
        
        self.device = None
        
        # relabeling을 위해 
        # self.episodes가 있음. ep_index만 저장하면 된다. 
        # popleft 할 때 마다 -> 전체 ep_index 하나씩 빼주면 됨. 
        # 바뀜.

        # transition 만들 때 ep index 하나 만들고
        # ep 하나 추가할 때 마다 
        # 0에서 시작 -> popleft 안하면 +1 

    @property
    def goal(self):
        return self.transitions[:, self.layout['goal']]

    @property
    def ep_index(self):
        return self.transitions[:, self.layout['ep_index']]


    def ep_to_transtions(self, episode):
        len_ep = len(episode.states[:-1])

        # last state = goal로 변경
        # relabeled reward 추가

        return torch.as_tensor(np.concatenate([
            episode.states[:-1],
            episode.actions,
            np.array(episode.rewards)[:, None],
            np.array(episode.dones)[:, None],
            episode.states[1:],
            episode.goals,
            np.array(list(range(len(episode.states) - 1)))[:, None]
        ], axis=-1))

    def enqueue(self, episode):
        # ep_index 로직
        # 일단 0에서 시작
        # episodes에서 제거 안했으면 -> ep_index += 1
        # 제거 했으면 -> 

        ep_delta = 0
        # 에피소드 길이가 0 이 될 때 까지
        while len(self.episodes) > 0:
            # 가장 앞의 에피소드와
            old_episode = self.episodes[0]
            # 그 ptr을 가져옴
            ptr = self.episode_ptrs[0]
            # ptr - self.ptr을 최대크기 (20000) 으로 나눈 몫임. 
            dist = (ptr - self.ptr) % self.max_size

            # 꽉 찰때 까지는 dist가 max_size부터 0까지 계속 작아짐.
            # 따라서 이 조건에 안걸림.
            # 꽉 찰 즈음 되면 이 조건에 걸린다. 
            if dist < len(episode):
                self.episodes.popleft()
                self.episode_ptrs.popleft()
                # 뺄 때 마다 ep_index도 변경 
                ep_delta +=  1
            else:
                break

        ep_index = len(self.episodes) - 1

        self.episodes.append(episode)
        self.episode_ptrs.append(self.ptr)
        self.transitions[:, self.layout['ep_index']] -= ep_delta

        transitions = self.ep_to_transtions(episode)
        transitions = torch.cat((transitions, torch.full((transitions.shape[0], 1), ep_index)), dim = -1)

        
        # self.ptr + 에피소드 길이가 최대 크기 이하
        # 즉, 처음부터 채우고 있는 과정임.
        if self.ptr + len(episode) <= self.max_size:
            # 빈 곳에 할당
            self.transitions[self.ptr:self.ptr+len(episode)] = transitions
        # 만약 1배 이상 2배 이하라면? 
        elif self.ptr + len(episode) < 2*self.max_size:
            # 잘라서 앞에넣고 뒤에넣고
            self.transitions[self.ptr:] = transitions[:self.max_size-self.ptr]
            self.transitions[:len(episode)-self.max_size+self.ptr] = transitions[self.max_size-self.ptr:]
        else:
            raise NotImplementedError

        # 즉, ptr은 현재 episode를 더하고 난 후의 위치임. 
        self.ptr = (self.ptr + len(episode)) % self.max_size
        self.size = min(self.size + len(episode), self.max_size)

    def sample(self, n):
        indices = torch.randint(self.size, size=[n])
        transitions = self.transitions[indices]
        batch = GC_Batch2(*[transitions[:, i] for i in self.layout.values()], transitions, self.tanh, self.hindsight_relabel).to(self.device).parse()
        
        relabeled_goals = []
        relabeled_rewards = []
        drws = []
        relabeled_dones = []
        
        if self.cfg.binary_reward:
            if self.cfg.structure == "flat_wgcsl":
                # wgcsl은 항상 relabel 
                relabel = np.where(np.random.rand(len(batch.state_index)) > -1, True, False)
            else:
                relabel = np.where(np.random.rand(len(batch.state_index)) > 0.8, True, False)
            # ep index로 relabeling 
            for i, (state_index, ep_index) in enumerate(zip(batch.state_index, batch.ep_index)):
                state_index = int(state_index.detach().cpu().item())
                ep_index = int(ep_index.detach().cpu().item())
                ep = self.episodes[ep_index]

                if relabel[i]:
                    try:
                        goal_index = np.random.randint(state_index, len(ep.states) - 1, 1)[0]
                    except:
                        goal_index = -1

                    # goal relabeling 
                    relabeled_goal = self.state_processor.goal_transform(ep.states[goal_index])
                    relabeled_goals.append(relabeled_goal)
                    
                    # discounted relabeling weight for WGCSL 
                    drw = np.exp(np.log(0.99) * (goal_index - state_index))
                    drws.append(drw)

                    # reward relabeling 
                    if self.cfg.structure == "flat_wgcsl":
                        if goal_index - state_index < self.cfg.reward_threshold:
                            relabeled_rewards.append(np.array([1]))
                            relabeled_dones.append(1)
                        else:
                            relabeled_rewards.append(np.array([0]))
                            relabeled_dones.append(0)

                    elif self.cfg.structure == "flat_ris":
                        if goal_index - state_index < self.cfg.reward_threshold:
                            relabeled_rewards.append(1)
                            relabeled_dones.append(1)
                            
                        else:
                            relabeled_rewards.append(-1)
                            relabeled_dones.append(0)

                    else: # skill-based  
                        if goal_index - state_index < 1:
                            relabeled_rewards.append(1)
                            relabeled_dones.append(1)
                        else:
                            relabeled_rewards.append(0)
                            relabeled_dones.append(0)

                else:
                    relabeled_goals.append(batch.G[i].detach().cpu().numpy())
                    
                    rr = batch.rewards[i].detach().cpu().numpy()
                    
                    relabeled_rewards.append(batch.rewards[i].detach().cpu().item())
                    
                    goal_index = len(ep.states) - 1
            
                    drw = np.exp(np.log(0.99) * (goal_index - state_index))

                    drws.append(drw)
                    relabeled_dones.append(batch.done[i].detach().cpu().numpy())



            # goal로 바꿔줘야 함. 
            batch["G"] = torch.tensor(relabeled_goals, dtype = torch.float32).to(self.device)
                        
            batch['reward'] = torch.tensor(relabeled_rewards, dtype = torch.float32).to(self.device)


            
            batch['drw'] = torch.tensor(drws, dtype = torch.float32).to(self.device)
            batch['done'] = torch.tensor(relabeled_dones, dtype = torch.float32).to(self.device)


            if self.cfg.structure == "flat_ris":
                # subgoal sample 
                states =  self.transitions[:, self.layout['states']]
                subgoal_indices = torch.randint(self.size, size=[n])
                batch['subgoal'] = states[subgoal_indices]


        return batch


class GC_Batch2(Batch):
    def __init__(self, states, actions, rewards, dones, next_states, goals, state_index, ep_index, transitions=None, tanh = False, hindsight_relabel = False):
        # def   __init__(states, actions, rewards, dones, next_states, transitions=None):
        super().__init__(states, actions, rewards, dones, next_states, transitions)

        self.data['goals'] = goals
        self.data['state_index'] = state_index
        self.data['ep_index'] = ep_index

        self.tanh = tanh
        self.hindsight_relabel = hindsight_relabel

    @property
    def goals(self):
        return self.data['goals']
    
    @property
    def state_index(self):
        return self.data['state_index']
        
    @property
    def ep_index(self):
        return self.data['ep_index']
    
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
            rewards = self.rewards,
            G = self.goals,
            done = self.dones,
            state_index = self.state_index,
            ep_index = self.ep_index
        )
        if self.tanh:
            batch_dict['actions'], batch_dict['actions_normal'] = self.actions.chunk(2, -1)
        else:
            batch_dict['actions'] = self.actions

        return batch_dict
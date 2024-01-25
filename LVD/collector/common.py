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
    """
    Hierarchical episode for relabeling 
    """
    def __init__(self, init_state, env_name = None, max_reward = None):
        
        assert max_reward is not None, "NEED MAXREWARD"
        super().__init__(init_state, env_name)
        self.high_actions = []
        self.max_reward = max_reward
    
    def add_step(self, low_action, high_action, next_state, goal, reward, relabeled_reward, done, info):
        # MDP transitions
        super().add_step(low_action, next_state, goal, reward, relabeled_reward, done, info)
        self.high_actions.append(high_action)

    def as_high_episode(self):
        """
        """

        high_episode = Episode_RR(self.states[0])
        prev_t = 0
        
        # add transitions 
        for t in range(1, len(self)):
            if self.high_actions[t] is not None:
                high_episode.add_step(
                    self.high_actions[prev_t],
                    self.states[t],
                    self.goals[t],
                    sum(self.rewards[prev_t:t]),
                    sum(self.relabeled_rewards[prev_t:t]), # not used 
                    self.dones[t],
                    self.infos[t]
                )
                prev_t = t
        
        high_episode.add_step(
            self.high_actions[prev_t],
            self.states[-1],
            self.goals[t],
            sum(self.rewards[prev_t:]), 
            self.max_reward - sum(self.relabeled_rewards[prev_t:]),  # not used 
            self.dones[-1], 
            self.infos[-1]
        )
        high_episode.raw_episode = self

        # ----------------------------------------------------------------------------------------------- #
        high_episode_relabel = Episode_RR(self.states[0])
        prev_t = 0
    
        return high_episode, high_episode_relabel


class GC_Batch(Batch):
    def __init__(self, states, actions, rewards, relabeled_rewards, dones, next_states, goals, relabeled_goals, transitions=None, tanh = False, hindsight_relabel = False, cfg = None):
        super().__init__(states, actions, rewards, dones, next_states, transitions)
        
        self.cfg =cfg
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
        batch_dict = edict(
            states = self.states,
            next_states = self.next_states,
            rewards = self.rewards,
            G = self.goals,
            relabeled_G = self.relabeled_goals,
            relabeled_rewards = self.relabeled_rewards, 
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
        
        self.transitions = torch.empty(cfg.buffer_size, 2*cfg.state_dim + action_dim + cfg.n_goal * 2 + 3) # rwd, relabeled rwd, dones

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
        relabeled_goal = self.state_processor.goal_transform(np.array(episode.states[-1]))
        relabeled_goals = np.tile(relabeled_goal, (len(episode.states)-1, 1))

        relabeled_rewards = deepcopy(episode.relabeled_rewards)
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
        batch = GC_Batch(*[transitions[:, i] for i in self.layout.values()], transitions, self.tanh, self.hindsight_relabel, self.cfg).to(self.device)
        return batch.parse()
    
class GC_Buffer_Relabel(Buffer):
    """
    """
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


        self.transitions = torch.empty(cfg.buffer_size, 2*cfg.state_dim + action_dim + cfg.n_goal * 2 + 4) # rwd, relabeled rwd, dones


        dims = OrderedDict([
            ('state', cfg.state_dim),
            ('action', action_dim),
            ('reward', 1),
            ('relabled_reward', 1),
            ('done', 1),
            ('next_state', cfg.state_dim),
            ('goal', cfg.n_goal),
            ('relabled_goal', cfg.n_goal),
            ('drws', 1),
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
    def ep_index(self):
        return self.transitions[:, self.layout['ep_index']]


    def ep_to_transtions(self, episode):
        len_ep = len(episode.states[:-1])

        relabeled_rewards = deepcopy(episode.relabeled_rewards)
        

        relabeled_goals = []
        relabeled_rewards = []
        drws = []
        dones = []

        discount = np.log(0.99)
        
        for state_index in range(len(episode.states) - 1):
            relabel_index = np.random.randint(state_index, len_ep, 1)[0]
            
            # g_t \gets \Phi(s_t)
            _now_state_G = self.state_processor.goal_transform(np.array(episode.states[state_index]))
            
            # g_i \gets \Phi(s_i) where i > t 
            relabeled_goal = self.state_processor.goal_transform(np.array(episode.states[relabel_index]))
            
            if self.cfg.distance_reward:
                # no kitchen 
                assert self.cfg.env_name == "maze", f"Invalid env name : {self.cfg.env_name}"
                if np.linalg.norm(_now_state_G - relabeled_goal) < 1:
                    rr, done = 1, 1
                else:
                    rr, done = 1, 0
                
                
            else:
                if relabel_index - state_index < 2:
                    rr, done = 1, 1 
                else:
                    if self.cfg.structure == "flat_wgcsl":
                        rr = 0
                    elif self.cfg.structure == "flat_ris":
                        rr = -1
                    else:
                        rr = 0        
                    done = 0        

            relabeled_goals.append(relabeled_goal)
            relabeled_rewards.append(rr)
            drws.append(np.exp(discount * (relabel_index - state_index)))
            dones.append(done)            

        relabeled_goals = np.array(relabeled_goals)
        relabeled_rewards = np.array(relabeled_rewards)[:, None]
        drws = np.array(drws)[:, None]
        dones = np.array(dones)[:, None]

        return torch.as_tensor(np.concatenate([
            episode.states[:-1],
            episode.actions,
            np.array(episode.rewards)[:, None],
            relabeled_rewards,            
            dones, #np.array(episode.dones)[:, None],
            episode.states[1:],
            episode.goals,
            relabeled_goals,  
            drws,          
        ], axis=-1))

    def enqueue(self, episode):
        for _ in range(3):
            self.__enqueue__(episode) 

    def __enqueue__(self, episode):
        while len(self.episodes) > 0:
            old_episode = self.episodes[0]
            ptr = self.episode_ptrs[0]
            dist = (ptr - self.ptr) % self.max_size

            if dist < len(episode):
                self.episodes.popleft()
                self.episode_ptrs.popleft()
            else:
                break


        self.episodes.append(episode)
        self.episode_ptrs.append(self.ptr)

        transitions = self.ep_to_transtions(episode)

        

        if self.ptr + len(episode) <= self.max_size:
            self.transitions[self.ptr:self.ptr+len(episode)] = transitions
        elif self.ptr + len(episode) < 2*self.max_size:
            self.transitions[self.ptr:] = transitions[:self.max_size-self.ptr]
            self.transitions[:len(episode)-self.max_size+self.ptr] = transitions[self.max_size-self.ptr:]
        else:
            raise NotImplementedError

        self.ptr = (self.ptr + len(episode)) % self.max_size
        self.size = min(self.size + len(episode), self.max_size)

    def sample(self, n):
        indices = torch.randint(self.size, size=[n])
        transitions = self.transitions[indices]
        batch = GC_Batch2(*[transitions[:, i] for i in self.layout.values()], transitions, self.tanh, self.hindsight_relabel).to(self.device).parse()
        
        if self.cfg.structure == "flat_ris":
            # subgoal sample 
            states =  self.transitions[:, self.layout['state']]
            subgoal_indices = torch.randint(self.size, size=[n])
            batch['subgoals'] = states[subgoal_indices]
            
        elif self.cfg.structure in ['flat_gcsl', 'flat_wgcsl']:
            # always relabel
            batch['G'] = batch['relabeled_goal']            
            batch['reward'] = batch['relabeled_reward'].squeeze(-1)


        return batch


class GC_Batch2(Batch):
    def __init__(self, states, actions, rewards, relabeled_rewards, dones, next_states, goals, relabeled_goals, drws, transitions=None, tanh = False, hindsight_relabel = False):
        super().__init__(states, actions, rewards, dones, next_states, transitions)

        self.data['goals'] = goals
        self.data['relabeled_rewards'] = relabeled_rewards
        self.data['relabeled_goals'] = relabeled_goals
        self.data['drws'] = drws

        self.tanh = tanh
        self.hindsight_relabel = hindsight_relabel

    @property
    def goals(self):
        return self.data['goals']
    
    @property
    def relabeled_rewards(self):
        return self.data['relabeled_rewards']
        
    @property
    def relabeled_goals(self):
        return self.data['relabeled_goals']
    @property
    def drws(self):
        return self.data['drws']
    
    def parse(self):    
        batch_dict = edict(
            states = self.states,
            next_states = self.next_states,
            reward = self.rewards,
            relabeled_reward = self.relabeled_rewards,
            G = self.goals,
            done = self.dones,
            relabeled_goal = self.relabeled_goals,
            drw = self.drws
        )
        if self.tanh:
            batch_dict['actions'], batch_dict['actions_normal'] = self.actions.chunk(2, -1)
        else:
            batch_dict['actions'] = self.actions

        return batch_dict
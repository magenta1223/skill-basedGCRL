from ..contrib.simpl.collector.storage import Episode 
from ..utils import StateProcessor

# from ..contrib.simpl.collector.hierarchical import HierarchicalEpisode
# from ..utils import GOAL_CHECKERS

import numpy as np
from copy import deepcopy
import torch

from .storage import Episode_G

class HierarchicalEpisode_G(Episode_G):
    def __init__(self, init_state):
        super().__init__(init_state)
        self.low_actions = self.actions
        self.__high_actions__ = []


    def add_step(self, low_action, high_action, next_state, G, reward, done, info):
        # MDP transitions
        # def add_step(self, action, next_state, G, reward, done, info):
        super().add_step(low_action, next_state, G, reward, done, info)
        self.__high_actions__.append(high_action)

    def as_high_episode(self):
        """
        high-action은 H-step마다 값이 None 이 아
        """

        high_episode = Episode(self.states[0])
        prev_t = 0
        for t in range(1, len(self)):
            if self.high_actions[t] is not None:
                # high-action은 H-step마다 값이 None이 아니다.
                # raw episode를 H-step 단위로 끊고, action을 high-action으로 대체해서 넣음. 
                # add_step(self, action, next_state, G, reward, done, info):
                high_episode.add_step(
                    self.high_actions[prev_t], # skill
                    self.states[t], # next_state
                    self.G[t], # G
                    sum(self.rewards[prev_t:t]),
                    self.dones[t],
                    self.infos[t]
                )
                prev_t = t
        
        high_episode.add_step(
            self.high_actions[prev_t],
            self.states[-1],
            self.G[-1], # G
            sum(self.rewards[prev_t:]),
            self.dones[-1],
            self.infos[-1]
        )
        high_episode.raw_episode = self
        return high_episode
    
    @property
    def high_actions(self):
        return np.array(self.__high_actions__)


class HierarchicalTimeLimitCollector:
    def __init__(self, env, horizon, time_limit=None):
        self.env = env
        self.env_name = self.env.env_name
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf

        self.state_processor = StateProcessor(env_name= self.env_name)


    def collect_episode(self, low_actor, high_actor, verbose = True):
        state, done, t = self.env.reset(), False, 0

        # G = GOAL_TRANSFORM[self.env_name](state)
        # state = STATE_PROCESSOR[self.env_name](state)



        G = self.state_processor.get_goals(state)
        state = self.state_processor.state_process(state)

        print(f"G : {self.state_processor.goal_checker(G)}")

        episode = HierarchicalEpisode_G(state)
        low_actor.eval()
        high_actor.eval()
        
        while not done and t < self.time_limit:

            # print(goal_checker(state[30:]))
            if t % self.horizon == 0:
                high_action, high_action_normal = high_actor.act(state, G)
                data_high_action = high_action
            else:
                data_high_action = None
            
            # print(high_action.shape)
            
            with low_actor.condition(high_action):
                low_action = low_actor.act(state)


            state, reward, done, info = self.env.step(low_action)
            # state = STATE_PROCESSOR[self.env_name](state)
            state = self.state_processor.state_process(state)


            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']

            if data_high_action is not None:
                data_high_action_w_normal = np.concatenate((data_high_action, high_action_normal), axis = -1)
            else:
                data_high_action_w_normal = None
            
            episode.add_step(low_action, data_high_action_w_normal, state, G, reward, data_done, info)
            t += 1
            
                
    
        # print(GOAL_CHECKERS[self.env_name](STATE_PROCESSOR[self.env_name] (state)))
        if verbose:
            print( self.state_processor.state_goal_checker(state, self.env)  )
        
        # print(np.array(episode.actions)[:, 0].mean())
        # print(np.mean([info['vel'] for info in episode.infos]))
        # print(len(episode))


    

        return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)

    
class LowFixedHierarchicalTimeLimitCollector(HierarchicalTimeLimitCollector):
    def __init__(self, env, low_actor, horizon, time_limit=None):
        super().__init__(env, horizon, time_limit)
        self.low_actor = low_actor

    def collect_episode(self, high_actor):
        return super().collect_episode(self.low_actor, high_actor)

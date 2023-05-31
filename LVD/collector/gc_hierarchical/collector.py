from ...contrib.simpl.collector.hierarchical import HierarchicalEpisode, Episode
from ...utils import StateProcessor

import numpy as np
from copy import deepcopy
import torch

class Episode_RR(Episode):
    def __init__(self, init_state):
        super().__init__(init_state)
        self.relabeled_rewards = []
        self.goals = []

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



class GC_Hierarchical_Collector:
    def __init__(self, env, low_actor, horizon, time_limit=None):
        self.env = env
        self.env_name = env.name
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.state_processor = StateProcessor(env_name= self.env_name)

        self.low_actor = low_actor

    @torch.no_grad()
    def collect_episode(self, high_actor, verbose = True, vis = False):
        state, done, t = self.env.reset(), False, 0
        G = self.state_processor.get_goal(state)
        state = self.state_processor.state_process(state)

        # episode = HierarchicalEpisode(state)
        episode  =HierarchicalEpisode_RR(state)
        episode.goal = G
        self.low_actor.eval()
        high_actor.eval()
        imgs = []
        
        while not done and t < self.time_limit:
            if t % self.horizon == 0:
                high_action_normal, high_action = high_actor.act(state, G)
                data_high_action_normal, data_high_action = high_action_normal, high_action
            else:
                data_high_action = None
                        
            with self.low_actor.condition(high_action):
                low_action = self.low_actor.act(state)


            state, reward, done, info = self.env.step(low_action)
            relabeled_reward = info['relabeled_reward']
            state = self.state_processor.state_process(state)

            if vis:
                imgs.append(self.env.render(mode= "rgb_array"))


            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            
            if data_high_action is not None:
                data_high_action_w_normal = np.concatenate((data_high_action, data_high_action_normal), axis = 0)
            else:
                data_high_action_w_normal = None
            
            episode.add_step(low_action, data_high_action_w_normal, state, G, reward, relabeled_reward, data_done, info)


            t += 1
        if verbose:
            print( self.state_processor.state_goal_checker(state)  )

        if vis:
            return imgs
        else:
            return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)
# from ...contrib.simpl.collector.hierarchical import HierarchicalEpisode, Episode

from ..common import *
from ...utils import StateProcessor, coloring

import numpy as np
from copy import deepcopy
import torch


class GC_Hierarchical_Collector:
    def __init__(self, cfg, env, low_actor, horizon, time_limit=None):
        self.cfg = cfg
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
        # episode  =HierarchicalEpisode_RR(state)
        episode = HierarchicalEpisode_Relabel(state, self.env_name, self.cfg.binary_reward, self.cfg.max_reward)
        # episode.goal = G
        self.low_actor.eval()
        high_actor.eval()
        imgs = []
        
        while not done and t < self.time_limit:
            if t % self.horizon == 0:
                # if "act_cem" in  dir(high_actor):
                #     high_action_normal, high_action = high_actor.act_cem(state, G)
                # else:
                #     high_action_normal, high_action = high_actor.act(state, G)

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
        # if 'TimeLimit.truncated' in info:
        #     print(data_done, info['TimeLimit.truncated'])
        # else:
        #     print(data_done)
        if verbose:
            targetG = self.state_processor.state_goal_checker(G)
            achievedG = self.state_processor.state_goal_checker(state)
            coloring(self.env.name, targetG, achievedG, data_done, episode) 
            # print(f"T : {self.state_processor.state_goal_checker(G)}, A : {self.state_processor.state_goal_checker(state)}")

        if vis:
            return imgs
        else:
            return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)
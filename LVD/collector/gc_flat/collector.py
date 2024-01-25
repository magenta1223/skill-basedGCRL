from ...utils import StateProcessor, coloring

import numpy as np
import torch
from ..common import *



class GC_Flat_Collector:
    """
    Episode Collector for action-step methdos.
    """
    def __init__(self, env, time_limit=None):
        self.env = env
        self.env_name = env.name
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.state_processor = StateProcessor(env_name= self.env_name)
    
    @torch.no_grad()
    def collect_episode(self, actor, verbose = True, vis = False):
        state, done, t = self.env.reset(), False, 0
        G = self.state_processor.get_goal(state)
        state = self.state_processor.state_process(state)
        episode = Episode_RR(state)
        actor.eval()
        imgs = []
        while not done and t < self.time_limit:
            action_normal, action = actor.act(state, G)
            state, reward, done, info = self.env.step(action)
            relabeled_reward = info['relabeled_reward']
            state = self.state_processor.state_process(state)
            if vis:
                imgs.append(self.env.render(mode= "rgb_array"))
            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            if action_normal is not None:
                action = np.concatenate((action, action_normal))
            episode.add_step(action, state, G, reward, relabeled_reward, data_done, info)

            t += 1
        if verbose:

            targetG = self.state_processor.state_goal_checker(G)
            achievedG = self.state_processor.state_goal_checker(state)
            coloring(self.env.name, targetG, achievedG, data_done) 

        if vis:
            return imgs
        else:
            return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)
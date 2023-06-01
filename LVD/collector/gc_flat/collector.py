from ...contrib.simpl.collector.storage import Episode 
from ...utils import StateProcessor

import numpy as np
from copy import deepcopy
import torch
from ..common import *



class GC_Flat_Collector:
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

        # print(f"G : {self.state_processor.goal_checker(G)}")

        episode = Episode_RR(state)
        actor.eval()
        imgs = []
        
        while not done and t < self.time_limit:
            action = actor.act(state, G)

            state, reward, done, info = self.env.step(action)
            state = self.state_processor.state_process(state)

            if vis:
                imgs.append(self.env.render(mode= "rgb_array"))


            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            
            
            episode.add_step(action, state, reward, data_done, info)


            t += 1
        if verbose:
            print( self.state_processor.state_goal_checker(state)  )

        if vis:
            return imgs
        else:
            return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)
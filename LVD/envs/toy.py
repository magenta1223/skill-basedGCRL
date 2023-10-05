import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env
from contextlib import contextmanager
from ..contrib.simpl.env.kitchen import KitchenTask, KitchenEnv

from ..utils import StateProcessor
import gym


class Nav2DTask:
    def __init__(self, goal) -> None:
        self.goal = np.array(goal).astype(np.float64)
        
    def __repr__(self):
        return f'Nav2D:{self.goal}'
        
        

class Navigation2D(gym.Env):
    """
    """

    def __init__(self):
        super(Navigation2D, self).__init__()
        
        self.name = "Nav2D"
        
        self.state = np.array([0.0 ,0.0])
        self.goal = np.array([0.0, 0.0])
        self.n_action_step = 0
        self.timelimit = 200
        self.task = None
 
    def step(self, action):
        # return super().step(action)
        # action은 최대 0.1
        
        self.n_action_step += 1
        action = np.clip(action, -0.1, 0.1)
        
            
        self.state += action         
        # reward 
        if np.linalg.norm(self.state -self.goal) < 0.1:
            reward = 1
            done = True
        else:
            reward = -1
            done = False
            
        if not done and self.n_action_step == self.timelimit:
            done = True
        
        env_info = {}
        env_info['relabeled_reward'] = 0 
        env_info['orig_return'] = reward    
        
        obs = np.concatenate((self.state, self.goal), axis = -1)
                
        return obs, reward, done, env_info
    
    def reset(self, *, seed= None, return_info = False, options = None):
        # return super().reset(seed=seed, return_info=return_info, options=options)
        # reinit
        self.state = np.array([0.0 ,0.0])
        
        if self.task is not None:
            self.goal = self.task.goal
        else:
            self.goal = np.array([999.0, 999.0])        
        
        obs = np.concatenate((self.state, self.goal), axis = -1)
        return obs
    
    # def render(self, mode="human"):
    #     return super().render(mode)
    
    @contextmanager
    def set_task(self, task):
        if type(task) != Nav2DTask:
            raise TypeError(f'task should be Nav2DTask but {type(task)} is given')

        prev_goal = self.goal
        self.goal = task.goal        
        self.task = task
        yield
        self.task = None
        self.goal = prev_goal

        
    def __repr__(self):
        return f'Nav2D'


zeroshot_tasks = np.array([
    # seen 

])

few_shot_tasks = np.array([
    # seen 
    [0,1],
    [0.3, 0.8],
    [0.1, 0.9],
    [-0.7, 0.2],
    [-2, 1]
])



toy_cfg = {}
toy_zeroshot_tasks = zeroshot_tasks
toy_fewshot_tasks = few_shot_tasks
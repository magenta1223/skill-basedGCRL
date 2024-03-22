import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env
from contextlib import contextmanager
# from ..contrib.simpl.env.kitchen import KitchenTask, KitchenEnv
from d4rl.locomotion.ant import AntMazeEnv
from ..utils import StateProcessor
from d4rl.utils.wrappers import NormalizedBoxEnv
from d4rl.locomotion import maze_env
from collections import deque 



mujoco_env.USE_DM_CONTROL = False
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


class AntMazeTask_GC:
    def __init__(self, location):
        self.location = location[:2] # except for optimal distance 

    def __repr__(self):
        return f'{self.location}'

# class AntMazeEnv_GC(AntMazeEnv):
class AntMazeEnv_GC(NormalizedBoxEnv):
    """
    Goal Conditioned Environment
    """
    # render_width = 400
    # render_height = 400
    # render_device = -1
    name = "antmaze"
    
    def __init__(self, *args, **kwargs):
        # super().__init__(**kwargs)
        super().__init__(AntMazeEnv(**kwargs))
        self.task = None

        
    @contextmanager
    def set_task(self, task):
        if type(task) != AntMazeTask_GC:
            raise TypeError(f'task should be AntMazeTask_GC but {type(task)} is given')

        prev_task = self.task
        self.task = task
        # self.set_target(self.task.location)
        self._wrapped_env.set_target(self.task.location)
        yield
        self.task = prev_task
    
    def reset(self):
        return self.reset_model()

    def reset_model(self):
        # obs = super().reset_model()
        obs = self.wrapped_env.reset_model()
        return np.concatenate((obs, self._goal), axis = -1)

    def step(self, a):
        # obs, reward, done, env_info = super().step(a)
        obs, reward, done, env_info = super().step(a)        
        # concatenate goal 
        obs = np.concatenate((obs, self._goal), axis = -1)
        return obs, reward, done, env_info



# seen 
# [(4.42, 0.41, 1), 
#  (0.24, 4.14, 1), 
#  (0.2, 8.09, 2), 
#  (8.49, 0.38, 2), 
#  (4.1, 8.47, 3), 
#  (0.38, 12.26, 3), 
#  (12.28, 0.09, 3), 
#  (0.3, 16.2, 4), 
#  (4.4, 16.22, 5), 
#  (16.25, 8.29, 6), 
#  (20.29, 8.39, 7), 
#  (4.47, 24.0, 7), 
#  (20.18, 12.45, 8), 
#  (0.47, 24.01, 8), 
#  (20.09, 16.1, 9), 
#  (20.28, 20.21, 10), 
#  (20.21, 24.22, 11), 
#  (16.4, 24.22, 12), 
#  (12.42, 24.16, 13), 
#  (12.19, 20.04, 14), 
#  (12.15, 16.19, 15)]


# unseen 
# [
#     (12.45, 4.16, 4), 
#     (8.2, 8.08, 4), 
#     (12.14, 8.09, 5), 
#     (4.28, 20.03, 6), 
#     (20.22, 4.2, 8), 
#     (20.24, 0.27, 9), 
#     (24.25, 0.05, 10),
#     (24.01, 16.39, 10), 
#     (28.36, 0.38, 11), 
#     (28.06, 16.38, 11), 
#     (32.17, 0.1, 12), 
#     (28.13, 4.05, 12), 
#     (32.19, 16.18, 12), 
#     (28.21, 20.23, 12), 
#     (36.01, 0.08, 13), 
#     (28.06, 8.43, 13), 
#     (36.18, 16.06, 13), 
#     (28.41, 24.09, 13), 
#     (36.27, 4.21, 14), 
#     (32.34, 8.05, 14), 
#     (36.16, 12.39, 14), 
#     (32.27, 24.48, 14), 
#     (36.17, 8.43, 15), 
#     (36.01, 24.23, 15)
# ]



zeroshot_tasks = np.array([
    # seen
    [4, 0],
    [0, 4],
    [0, 8],
    [8, 0],
    # [4, 8],
    # [0, 12],
    # [12, 0],
    # [0, 16],
    # [4, 16],
    # [16, 8],
    # [20, 8],
    # [4, 24],
    # small
    [12, 4],
    [8, 8],
    [12, 8],
    # medium
    [4, 20],
    [20, 4],
    [20, 0],
    # [24, 0],
    # [24, 16],
    # large
    [28, 0],
    [28, 16],
    [32, 0],
    # [28, 4],
    # [32, 16],
    # [28, 20],
    # [36, 0],
    # [28, 8],
    # [36, 16],
    # [28, 24],
    # [36, 4],
    # [32, 8],
    # [36, 12],
    # [32, 24],
    # [36, 8],
    # [36, 24],
])

few_shot_tasks = np.array([
    # large

])



antmaze_cfg={
    'deprecated': True,
    'maze_map': maze_env.HARDEST_MAZE_TEST,
    'reward_type':'sparse',
    'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'non_zero_reset':False, 
    'eval':True,
    'maze_size_scaling': 4.0,
    'ref_min_score': 0.0,
    'ref_max_score': 1.0,
}



antmaze_zeroshot_tasks = zeroshot_tasks
antmaze_fewshot_tasks = few_shot_tasks
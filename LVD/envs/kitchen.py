import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env
from contextlib import contextmanager
from ..contrib.simpl.env.kitchen import KitchenTask, KitchenEnv

from ..utils import StateProcessor

mujoco_env.USE_DM_CONTROL = False
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


class KitchenTask_GC(KitchenTask):
    def __repr__(self):
        return "".join([ t[0].upper() for t in self.subtasks])

class KitchenEnv_GC(KitchenEnv):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1
    name = "kitchen"
    
    def __init__(self, binary_reward = False, *args, **kwargs):
        self.TASK_ELEMENTS = ['top burner']  # for initialization
        self.ALL_SUBTASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
        self.all_subtasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


        self.binary_reward = binary_reward

        super().__init__(*args, **kwargs)
        
        self.task = None
        self.TASK_ELEMENTS = None
    
    def _get_task_goal(self, task=None):
        if task is None:
            task = self.TASK_ELEMENTS
                    
        # initial state 
        new_goal = np.array([ 
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , # robot arms
            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,  0.  ,  0.  ,
            0.  , 0.  , 0.01, 0.  ,  0.  , 0.27,  0.35,  1.62,  1.  ,
            0.  , 0.  , 0.  ])

        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal
    
    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask_GC:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks
        self.all_subtasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
        
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements
    
    def step(self, a):
        obs, reward, done, env_info = super().step(a)
        
        # for logging when binary reward 
        env_info['orig_return'] = 4 - len(self.tasks_to_complete)

        if self.binary_reward:
            reward = 1 if len(self.tasks_to_complete) == 0 else 0
            
        return obs, reward, done, env_info

zeroshot_tasks = np.array([
    # seen 
    [5,6,0,3], # MKBS
    [5,6,3,4], # MKSH
    [6,1,2,3], # KTLS
    [6,0,1,4], # KBTH
    
    # small
    [5,6,0,1], # MKBT
    [5,6,0,2], # MKBL
    [5,0,2,4],  # MBLH

    # middle
    [0,2,3,4],  # BLSH
    [6,1,3,4],  # KTSH
    [5,6,1,3],  # MKTS
    [5,1,2,3],  # MTLS
    
    # large
    [6,1,2,4],  # KTLH
    [5,1,3,4],  # MTSH
    [0,1,2,4],  # BTLH
])

few_shot_tasks = np.array([
    # large
    [6,1,2,4],  # KTLH
    [5,1,3,4],  # MTSH
    [0,1,2,4],  # BTLH
])




kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])

kitchen_cfg = {}
kitchen_zeroshot_tasks = kitchen_subtasks[zeroshot_tasks]
kitchen_fewshot_tasks = kitchen_subtasks[few_shot_tasks]
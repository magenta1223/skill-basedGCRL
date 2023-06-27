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
        return "-".join([ t[0].upper() for t in self.subtasks])

class KitchenEnv_GC(KitchenEnv):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1
    name = "kitchen"

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = ['top burner']  # for initialization
        self.ALL_SUBTASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
        self.all_subtasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

        
        super().__init__(*args, **kwargs)
        

        self.task = None
        self.TASK_ELEMENTS = None
    
    def _get_task_goal(self, task=None):
        if task is None:
            task = self.TASK_ELEMENTS
        # new_goal = np.zeros_like(self.goal) # 여기서 zero가 아니고 초기 상태여야 함. 
        
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

        # print(f"TASK : {str(task)}")
        
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements

    def compute_relabeled_reward(self, obs_dict):
        reward_dict = {}
        
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = self._get_task_goal(task=self.ALL_SUBTASKS)
        idx_offset = len(next_q_obs)
        completions = []

        for element in self.all_subtasks:
            # 뭔가 완료 될 때 마다 reward에 +1 
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])

            complete = distance < BONUS_THRESH
            if complete:
                completions.append(element)

        for completion in completions:
            self.all_subtasks.remove(completion)
        relabeled_reward = float(len(completions))
                
        return relabeled_reward
    
    def step(self, a):
        obs, reward, done, env_info = super().step(a)
        relabeled_reward = self.compute_relabeled_reward(self.obs_dict)
        env_info['relabeled_reward'] = relabeled_reward
        return obs, reward, done, env_info

# for simpl
meta_train_tasks = np.array([
    [5,6,0,3],
    [5,0,1,3],
    [5,1,2,4],
    [6,0,2,4],
    [5,0,4,1],
    [6,1,2,3],
    [5,6,3,0],
    [6,2,3,0],
    [5,6,0,1],
    [5,6,3,4],
    [5,0,3,1],
    [6,0,2,1],
    [5,6,1,2],
    [5,6,2,4],
    [5,0,2,3],
    [6,0,1,2],
    [5,2,3,4],
    [5,0,1,4],
    [6,0,3,4],
    [0,1,3,2],
    [5,6,2,3],
    [6,0,1,4],
    [0,1,2,3]
])


tasks = np.array([
    # # Well-aligned, Not missing
    # [5,6,0,3], # MKBS
    # [5,0,1,3],  # MBTS
    # # Mis-aligned, Not missing
    # [6,0,2,4],  # KBLH
    # [5,1,2,4],  # MTLH
    # Well-algined, Missing
    [5,6,0,1], # MKBT
    [5,6,0,2], # MKBL
    # Mis-algined, Missing 
    [6,1,2,4],  # KTLH
    [5,1,3,4],  # MTSH
    # [5,6,0,2],  # MKBL
])


kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])

kitchen_cfg = {}
kitchen_tasks = kitchen_subtasks[tasks]
kitchen_meta_tasks = kitchen_subtasks[meta_train_tasks]
kitchen_ablation_tasks = kitchen_subtasks[tasks]



kitchen_known_tasks = ['KBTS','MKBS','MKLH','KTLS',
               'BTLS','MTLH','MBTS','KBLH',
               'MKLS','MBSH','MKBH','KBSH',
               'MBTH','BTSH','MBLS','MLSH',
               'KLSH','MBTL','MKTL','MKSH',
               'KBTL','KBLS','MKTH','KBTH']
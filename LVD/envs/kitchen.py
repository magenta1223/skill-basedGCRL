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


        env_info['orig_return'] = 4 - len(self.tasks_to_complete)

        if self.binary_reward:
            # reward = 1 if done else 0 
            reward = 1 if len(self.tasks_to_complete) == 0 else 0
            # tasks_to_complete가 없어야 0
            relabeled_reward = 0 # high-ep로 변환 시 마지막에 1추가 
            
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
    # seen 
    # [5,6,0,3], # MKBS
    # [5,6,3,4], # MKSH
    # [6,1,2,3], # KTLS
    # [6,0,1,4], # KBTH

    # 'MKBT' : "well-aligned",
    # 'MKBL' : "well-aligned",
    # 'BLSH' : "well-aligned",
    # 'MBLH' : "well-aligned",
    # 'KTSH' : "well-aligned",
    
    # small
    [5,6,0,1], # MKBT
    [5,6,0,2], # MKBL
    [0,2,3,4],  # BLSH

    # # middle
    # [5,0,2,4],  # MBLH
    # [6,1,3,4],  # KTSH
    # [5,6,1,3],  # MKTS
    # [5,1,2,3],  # MTLS
    
    # # large
    # [6,1,2,4],  # KTLH
    # [5,1,3,4],  # MTSH
    # [0,1,2,4],  # BTLH
    

    # "MKBT": "small",
    # "MKBL": "small",
    # "BLSH": "small",
    # "MBLH": "middle",
    # "KTSH": "middle",
    # "MKTS": "middle",
    # "MTLS": "middle",
    # "KTLH": "large",
    # "MTSH": "large",
    # "BTLH": "large"
])




zeroshot_tasks = np.array([
    # seen 
    # [5,6,0,3], # MKBS
    # [5,6,3,4], # MKSH
    # [6,1,2,3], # KTLS
    # [6,0,1,4], # KBTH
    

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
    # seen 
    # [5,6,0,3], # MKBS
    # [5,6,3,4], # MKSH
    # [6,1,2,3], # KTLS
    # [6,0,1,4], # KBTH
    

    # # small
    # [5,6,0,1], # MKBT
    # [5,6,0,2], # MKBL
    # [5,0,2,4],  # MBLH

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




kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])

kitchen_cfg = {}
kitchen_tasks = kitchen_subtasks[tasks]
kitchen_zeroshot_tasks = kitchen_subtasks[zeroshot_tasks]
kitchen_fewshot_tasks = kitchen_subtasks[few_shot_tasks]

kitchen_meta_tasks = kitchen_subtasks[meta_train_tasks]
kitchen_ablation_tasks = kitchen_subtasks[tasks]



kitchen_known_tasks = [
                    'KBTS','MKBS','MKLH','KTLS',
                    'BTLS','MTLH','MBTS','KBLH',
                    'MKLS','MBSH','MKBH','KBSH',
                    'MBTH','BTSH','MBLS','MLSH',
                    'KLSH','MBTL','MKTL','MKSH',
                    'KBTL','KBLS','MKTH','KBTH'
                    ]




kitchen_unseen_tasks = [
    'MKBT',
    'MKBL',
    'KTLH',
    'MTSH',
    'BLSH',
    'BTLH',
    'KTSH',
    'MBLH',
    'MKTS',
    'MTLS'
    ]
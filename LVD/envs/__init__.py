from .kitchen import *
from .carla import *
from .maze import *


class Kitchen_EnvTaskConfig:
    name = "kitchen"
    # env
    env_cls = KitchenEnv_GC
    env_cfg = kitchen_cfg
    # task 
    task_cls = KitchenTask_GC
    
    # tasks 
    meta_tasks = kitchen_meta_tasks
    target_tasks = kitchen_tasks
    ablation_tasks = kitchen_ablation_tasks
    
    # 
    zeroshot_tasks = kitchen_zeroshot_tasks
    fewshot_tasks = kitchen_fewshot_tasks
    
    known_tasks = kitchen_known_tasks
    unknown_tasks = kitchen_unseen_tasks

class Maze_EnvTaskConfig:
    name = "maze"
    # env 
    env_cls = Maze_GC
    env_cfg = maze_cfg
    # task 
    task_cls = MazeTask_GC
    
    # tasks 
    meta_tasks = maze_meta_tasks
    target_tasks = maze_tasks
    ablation_tasks = maze_ablation_tasks
    
    # 
    zeroshot_tasks = maze_zeroshot_tasks
    fewshot_tasks = maze_fewshot_tasks   
    
    known_tasks = None
    unknown_tasks = None

class Carla_EnvTaskConfig:
    name = "carla"
    env_cls = CARLA_GC
    env_cfg = carla_cfg
    task_cls = CARLA_Task
    meta_tasks = CARLA_META_TASKS
    target_tasks = CARLA_TASKS
    ablation_tasks = maze_ablation_tasks
    known_tasks = None
    unknown_tasks = None


__all__ = [
    'Kitchen_EnvTaskConfig',
    'Maze_EnvTaskConfig',
    'CARLA_EnvTaskConfig'

]

# class Carla_EnvTaskConfig:
#     env_cls = KitchenEnv_GC
#     env_cfg = kitchen_cfg
#     task_cls = KitchenTask_GC
#     meta_tasks = kitchen_meta_tasks
#     target_tasks = kitchen_tasks
#     ablation_tasks = kitchen_ablation_tasks


# ENV_TASK = {
#     "kitchen" : {
#         "env_cls" : KitchenEnv_GC,
#         "task_cls" : KitchenTask_GC,
#         "tasks" : KITCHEN_TASKS,
#         "cfg" : None ,
#         "ablation_tasks" : MAZE_ABLATION_TASKS
#     },
#     # "carla" : {
#     #     "env_cls" : CARLA_GC,
#     #     "task_cls" : CARLA_Task,
#     #     "tasks"  : CARLA_TASKS,
#     #     "cfg" : carla_config ,
#     #     "ablation_tasks" : MAZE_ABLATION_TASKS
#     # },
#     "maze" : {
#         "env_cls" : Maze_GC,
#         "task_cls" : MazeTask_Custom, 
#         "tasks"  : MAZE_TASKS,
#         "cfg" : maze_config ,
#         "ablation_tasks" : MAZE_ABLATION_TASKS
#     },
# }
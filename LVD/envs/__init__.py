from .kitchen import *
from .carla import *
from .maze import *
from .toy import *


class Kitchen_EnvTaskConfig:
    name = "kitchen"
    # env
    env_cls = KitchenEnv_GC
    env_cfg = kitchen_cfg
    # task 
    task_cls = KitchenTask_GC
    zeroshot_tasks = kitchen_zeroshot_tasks
    fewshot_tasks = kitchen_fewshot_tasks
    

class Maze_EnvTaskConfig:
    name = "maze"
    # env 
    env_cls = Maze_GC
    env_cfg = maze_cfg
    # task 
    task_cls = MazeTask_GC
    zeroshot_tasks = maze_zeroshot_tasks
    fewshot_tasks = maze_fewshot_tasks   

# class Carla_EnvTaskConfig:
#     name = "carla"
#     env_cls = CARLA_GC
#     env_cfg = carla_cfg
#     task_cls = CARLA_Task
#     meta_tasks = CARLA_META_TASKS
#     target_tasks = CARLA_TASKS
#     ablation_tasks = maze_ablation_tasks
#     known_tasks = None
#     unknown_tasks = None


# class Nav2D_EnvTaskConfig:
#     name = "2DNavigation"
#     env_cls = Navigation2D
#     env_cfg = toy_cfg
#     task_cls = Nav2DTask
#     zeroshot_tasks = toy_zeroshot_tasks
#     fewshot_tasks = toy_fewshot_tasks
    

__all__ = [
    'Kitchen_EnvTaskConfig',
    'Maze_EnvTaskConfig',
    'CARLA_EnvTaskConfig',
    'Nav2D_EnvTaskConfig'

]

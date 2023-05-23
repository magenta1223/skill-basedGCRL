from .kitchen import *
# from .carla import *
from .maze import *


class Kitchen_EnvTaskConfig:
    env_cls = KitchenEnv_GC
    env_cfg = kitchen_cfg
    task_cls = KitchenTask_GC
    meta_tasks = kitchen_meta_tasks
    target_tasks = kitchen_tasks
    ablation_tasks = kitchen_ablation_tasks

class Maze_EnvTaskConfig:
    env_cls = Maze_GC
    env_cfg = maze_cfg
    task_cls = MazeTask_GC
    meta_tasks = maze_meta_tasks
    target_tasks = maze_tasks
    ablation_tasks = maze_ablation_tasks


__all__ = [
    'Kitchen_EnvTaskConfig',
    'Maze_EnvTaskConfig'
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
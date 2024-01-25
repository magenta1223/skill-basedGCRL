from .kitchen import *
from .maze import *

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

__all__ = [
    'Kitchen_EnvTaskConfig',
    'Maze_EnvTaskConfig',
]
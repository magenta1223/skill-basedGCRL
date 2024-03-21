from .kitchen import *
from .maze import *
from .ant import * 


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


class Antmaze_EnvTaskConfig:
    name = "antmaze"
    # env
    env_cls = AntMazeEnv_GC
    env_cfg = antmaze_cfg
    # task 
    task_cls = AntMazeTask_GC
    zeroshot_tasks = antmaze_zeroshot_tasks
    fewshot_tasks = antmaze_fewshot_tasks


__all__ = [
    'Kitchen_EnvTaskConfig',
    'Maze_EnvTaskConfig',
    'Antmaze_EnvTaskConfig'
]
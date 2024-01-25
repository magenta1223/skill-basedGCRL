from contextlib import contextmanager

# from d4rl.pointmaze import MazeEnv
import mujoco_py
import numpy as np

from ..contrib.simpl.env.maze_layout import rand_layout
from ..contrib.simpl.env.maze import MazeEnv
from copy import deepcopy

init_loc_noise = 0.1
complete_threshold = 1.0

color_dict = {
    "wall" : np.array([0.87, 0.62, 0.38]),
    "agent" : np.array([0.32, 0.65, 0.32]),
    "ground_color1" : np.array([0.2, 0.3, 0.4]),
    "ground_color2" : np.array([0.1, 0.2, 0.3]),
}

WALL = np.full((32, 32, 3), color_dict['wall'])
G1 = np.full((32, 32, 3), color_dict['ground_color1'])
G2 = np.full((32, 32, 3), color_dict['ground_color2'])


class MazeTask_GC:
    def __init__(self, locs):
        init_loc, goal_loc = locs
        self.init_loc = np.array(init_loc, dtype=np.float32)
        self.goal_loc = np.array(goal_loc, dtype=np.float32)

    def __repr__(self):
        return f'{self.goal_loc}'

class Maze_GC(MazeEnv):
    name = "maze"

    def __init__(self, size, seed, reward_type, done_on_completed, relative = False, visual_encoder = None, *args, **kwargs):
        if reward_type not in self.reward_types:
            raise f'reward_type should be one of {self.reward_types}, but {reward_type} is given'
        # self.viewer_setup()
        self.size = size
        self.relative = relative
        super().__init__(size, seed, reward_type, done_on_completed)
        self.agent_centric_res = 32
        self.render_width = 32
        self.render_height = 32
        # for initialization
        self.task = MazeTask_GC([[0, 0], [0, 0]])
        self._viewers = {}
        self.viewer = self._get_viewer(mode = "rgb_array")

    @contextmanager
    def set_task(self, task):
        if type(task) != MazeTask_GC:
            raise TypeError(f'task should be MazeTask but {type(task)} is given')

        prev_task = self.task
        prev_target = self._target

        self.task = task
        self.set_target(task.goal_loc)
        print(f"TASK : {str(task)}")

        yield
        self._target = prev_target
        self.task = prev_task

    def reset_model(self):
        if self.task is None:
            raise RuntimeError('task is not set')
        init_loc = self.task.init_loc
        qpos = init_loc + self.np_random.uniform(low=-init_loc_noise, high=init_loc_noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        ob = deepcopy(self._get_obs())
        target = deepcopy(self._target)


        ob = np.concatenate((ob, target), axis = 0)
       
        return ob

    @contextmanager
    def agent_centric_render(self):
        prev_type = self.viewer.cam.type
        prev_distance = self.viewer.cam.distance
        
        self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
        self.viewer.cam.distance = 5.0
        
        yield
        
        self.viewer.cam.type = prev_type
        self.viewer.cam.distance = prev_distance
        


    def step(self, action):
        if self.task is None:
            raise RuntimeError('task is not set')
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()

        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = deepcopy(self._get_obs())

        goal_dist = np.linalg.norm(ob[0:2] - self._target)
        completed = (goal_dist <= complete_threshold)
        done = self.done_on_completed and completed
        
        target = deepcopy(self._target)

        ob = np.concatenate((ob, target), axis = 0)

        if self.reward_type == 'sparse':
            reward = float(completed) * 100
        elif self.reward_type == 'dense':
            reward = np.exp(-goal_dist)
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
                
        env_info = {}
        env_info['relabeled_reward'] = 0 

        env_info['orig_return'] = reward

        return ob, reward, done, env_info

    def compute_relabeled_reward(self):
        if self.reward_type == 'sparse':
            reward = 100
        elif self.reward_type == 'dense':
            reward = 1
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)

        return reward

    def render(self, mode = "rgb_array"):
        if mode == "agent_centric":
            with self.agent_centric_render():
                img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device) / 255
                walls = np.abs(img - WALL).mean(axis=-1)
                grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
            return img * 255

        else:
            return super().render(mode)
    
maze_cfg = {
    'size':40,
    'seed': 0,
    'reward_type':'sparse',
    'done_on_completed': True,
    'visual_encoder' : None
}


maze_zeroshot_tasks = np.array([
    # None 
    [[10, 24], [5, 25]], # 19
    [[10, 24], [1, 21]], # 18
    [[10, 24], [13, 26]], # 19    
    
    # Short 
    [[10, 24], [22, 23]], # 19
    [[10, 24], [1, 17]], # 18
    [[10, 24], [13, 9]], # 19

    # Middle 
    [[10, 24], [23, 14]], # 29
    [[10, 24], [18, 8]],  # 36
    [[10, 24], [24, 34]], # 32
    [[10, 24], [24, 39]], # 37
    [[10, 24], [15, 40]], # 40

    # extended

    [[10, 24], [36,  21]], # 47 
    [[10, 24], [39,  26]], # 45 
    [[10, 24], [15,  1]], # 45 
    [[10, 24], [25,  2]], # 45 
    ])

maze_fewshot_tasks = np.array([
    # extended

    [[10, 24], [36,  21]], # 47 
    [[10, 24], [39,  26]], # 45 
    [[10, 24], [15,  1]], # 45 
    [[10, 24], [25,  2]], # 45 
    ])


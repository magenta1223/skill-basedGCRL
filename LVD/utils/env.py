import numpy as np
import torch
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES, BONUS_THRESH

# --------------------------- Env, Model Utils --------------------------- #

def prep_state(states, device):
    """
    Correcting shape, device for env-model interaction
    """
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype = torch.float32)

    if len(states.shape) < 2:
        states = states.unsqueeze(0)

    states = states.to(device)
    return states
    

## ------- goal checker ------- ## 

def goal_checker_kitchen(state):
    achieved_goal = []
    for obj, indices in OBS_ELEMENT_INDICES.items():
        g = OBS_ELEMENT_GOALS[obj]
        distance = np.linalg.norm(state[indices] -g)   
        if distance < BONUS_THRESH:
            achieved_goal.append(obj)
    task = ""

    for sub_task in ['microwave', 'kettle', 'bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet']:
        if sub_task in achieved_goal:
            task += sub_task[0].upper() 
    return task


def goal_checker_maze(state):
    return (state[:2] * 1).astype(int)

def goal_checker_carla(state):
    return (state[-3:]).astype(int)

def goal_checker_calvin():
    return 

## ------- get goal state from state ------- ## 


def get_goal_kitchen(state):
    return state[30:]

def get_goal_maze(state):
    return state[-2:]

def get_goal_calvin():
    return 


def get_goal_carla(state):
    return state[-3 :-1] # only x, y

## ------- s ------- ## 


def goal_transform_kitchen(state):
    state[:9] = 0
    return state[:30]


def goal_transform_maze(state):
    return state[:2]

def goal_transform_calvin():
    return 

def goal_transform_carla(state):
    return state[12 :14]

## ------- env state -> obs ------- ## 

# SENSOR_SCALE = {
#     "control": (1, slice(0,3)),
#     "acceleration": (1, slice(0,3)),
#     "velocity": (10, slice(0,3)),
#     "angular_velocity": (10, slice(0,3)),
#     "location": (100, slice(0,2)),
#     "rotation": (10, slice(0,3)), # only steer 
#     "forward_vector": (1, slice(0,3)),  # remove
#     "target_location": (100, slice(0,0)), # remove 
# }


SENSOR_SCALE = {
    "control": (1, slice(0,3)),
    "acceleration": (1, slice(0,3)),
    "velocity": (1, slice(0,3)),
    "angular_velocity": (1, slice(0,3)),
    "location": (1/10, slice(0,2)),
    "rotation": (1/180, slice(0,3)), # only steer 
    "forward_vector": (1, slice(0,0)),  
    "target_location": (1, slice(0,0)), # remove 
}


SENSORS = ["control", "acceleration", "velocity", "angular_velocity", "location", "rotation", "forward_vector", "target_location"]


def state_process_kitchen(state):
    return state[:30]

def state_process_maze(state):
    return state[:-2]


def state_process_calvin():
    return 

def state_process_carla(state, normalize = False):
    if len(state.shape) == 2:
        obs_dict = { key : state[:, i*3 : (i+1)*3 ]   for i, key in enumerate(SENSORS)}
        prep_obs_dict = {}

        for k, (scale, idx) in SENSOR_SCALE.items():
            prep_obs_dict[k] = obs_dict[k][:, idx] * scale

            
            # contorl : all
            # acceleration : all
            # vel : all
            # angular vel : all
            # loc : all
            # rot : only y
            
        state = np.concatenate( [v for k, v in prep_obs_dict.items() if v.any()], axis = -1)
        return state
    else: 
        obs_dict = { key : state[i*3 : (i+1)*3 ]   for i, key in enumerate(SENSORS)}
        prep_obs_dict = {}

        for k, (scale, idx) in SENSOR_SCALE.items():
            prep_obs_dict[k] = obs_dict[k][idx]

            # raw_obs = obs_dict[k][idx] / scale
            # if raw_obs.
            # prep_obs_dict[k] = obs_dict[k][idx] / scale
            # print(k, prep_obs_dict[k])
            # contorl : all
            # acceleration : all
            # vel : all
            # angular vel : all
            # loc : all
            # rot : only y

        state = np.concatenate( [
            prep_obs_dict['control'],
            prep_obs_dict['acceleration'],
            prep_obs_dict['velocity'],
            prep_obs_dict['angular_velocity'],
            prep_obs_dict['location'],
            prep_obs_dict['rotation'],
            prep_obs_dict['forward_vector'],

        ], axis = -1)

        # state = np.concatenate( [v for k, v in prep_obs_dict.items() if v.any()], axis = -1)
        return state


    # xy = state[12:14] 
    # return np.concatenate((state[:12], xy, state[15:-6]), axis = -1)
    # return np.concatenate((state[:14], state[15:-6]), axis = -1) 
    # return state[:21]


class StateProcessor:

    def __init__(self, env_name):
        self.env_name = env_name

        self.__get_goals__ = {
            "kitchen" : get_goal_kitchen,
            "maze"    : get_goal_maze,
            "carla" : get_goal_carla
        }

        self.__goal_checkers__ = {
            "kitchen" : goal_checker_kitchen,
            "maze"  : goal_checker_maze,
            "carla" : goal_checker_carla

        }

        self.__state2goals__ = {
            "kitchen" : goal_transform_kitchen,
            "maze"  : goal_transform_maze,
            "carla"  : goal_transform_carla,

        }
        self.__state_processors__ = {
            "kitchen" : state_process_kitchen,
            "maze"  : state_process_maze,
            "carla" : state_process_carla
        }

    def get_goals(self, state):
        return self.__get_goals__[self.env_name](state)

    def goal_checker(self, goal):
        return self.__goal_checkers__[self.env_name] (goal)
    
    def state2goal(self, state):
        return self.__state2goals__[self.env_name](state)
    
    def state_process(self, state):
        return self.__state_processors__[self.env_name](state)
    
    def state_goal_checker(self, state, env, mode = "state"):
        """
        Check the state satisfiy which goal state
        """
        if self.env_name == "maze":
            if mode =="state":
                return self.__goal_checkers__[self.env_name](state) 
            else:
                return self.__goal_checkers__[self.env_name](state[-2:])

        if mode =="state":
            return self.__goal_checkers__[self.env_name](self.__state2goals__[self.env_name](state)) 
        else:
            return self.__goal_checkers__[self.env_name](self.__get_goals__[self.env_name](state)) 

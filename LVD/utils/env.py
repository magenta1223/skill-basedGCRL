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
    


class StateProcessor_Kitchen:
    @staticmethod
    def state_process(state):
        """
        Remove goal.
        """
        return state[..., :30]


    @staticmethod
    def goal_checker(state):
        """
        Check achieved goal.
        """
        assert len(state.shape) == 1, f"Only 1 sample for goal check. {state.shape}"

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

    @staticmethod
    def get_goal(state):
        """
        get goal state from state
        """
        return state[..., 30:]

    @staticmethod
    def goal_transform(state):
        """
        transform state to goal 
        """
        state[..., :9] = 0
        return state[..., :30]

    @staticmethod
    def to_ppc(state):
        """
        to proprioceptive state
        """
        state[..., 9:] = 0
        return state

class StateProcessor_Maze:
    @staticmethod
    def state_process(state):
        return state[..., :-2]

    @staticmethod
    def goal_checker(state):
        return (state[..., :2] * 1).astype(int)

    @staticmethod
    def get_goal(state):
        return state[..., -2:]

    @staticmethod
    def goal_transform(state):
        return state[..., :2]
    @staticmethod
    def to_ppc(state):
        """
        to proprioceptive state
        """
        state[..., 9:] = 0
        return state


class StateProcessor_AntMaze:
    @staticmethod
    def state_process(state):
        """
        remove goal 
        """
        return state[..., :-2]

    @staticmethod
    def goal_checker(state):
        return (state[..., :2] * 1).astype(int)

    @staticmethod
    def get_goal(state):
        return state[..., -2:]

    @staticmethod
    def goal_transform(state):
        return state[..., :2]
    
    @staticmethod
    def to_ppc(state):
        """
        to proprioceptive state
        """
        state[..., 9:] = 0
        return state


class StateProcessor_Toy:
    @staticmethod
    def state_process(state):
        return state[..., :2]

    @staticmethod
    def goal_checker(state):
        return state[...,-2:].astype(int)

    @staticmethod
    def get_goal(state):
        return state[...,-2:]

    @staticmethod
    def goal_transform(state):
        return state[...,:2]
    
    @staticmethod
    def to_ppc(state):
        """
        to proprioceptive state
        """
        return state



class StateProcessor:
    def __init__(self, env_name) -> None:
        processor_cls = dict(
            kitchen = StateProcessor_Kitchen,
            maze = StateProcessor_Maze,
            antmaze = StateProcessor_AntMaze,
            Nav2D = StateProcessor_Toy,
        )
        
        processor = processor_cls[env_name]()
        self.goal_checker = processor.goal_checker
        self.get_goal = processor.get_goal
        self.state_process = processor.state_process
        self.goal_transform = processor.goal_transform
        self.to_ppc = processor.to_ppc


    def state_goal_checker(self, state, mode = "state"):
        """
        Check the state satisfiy which goal state
        """
        if mode =="state":
            return self.goal_checker(self.goal_transform(state)) 
        else:
            return self.goal_checker(self.get_goal(state)) 
        

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    


def coloring(env_name, targetG, achievedG, done, ep = None):
    if ep is not None:
        reward = sum(ep.rewards)
    else:
        reward = ""

    if env_name == "kitchen":
        task_colors = ""
        for subT in achievedG:
            if subT not in targetG:
                task_colors += Colors.RESET + subT + Colors.RESET
            else:
                all_done_untilNow = sum([subT_todo not in achievedG for subT_todo in targetG[:targetG.index(subT)]]) == 0
                if all_done_untilNow:
                    task_colors += Colors.BLUE + subT + Colors.RESET
                else:
                    task_colors += Colors.RED + subT + Colors.RESET
        print(f"T : {targetG} A : {task_colors} R : {reward}")
    else:
        if done :
            print(f"T : {targetG} A : {Colors.BLUE}{achievedG}{Colors.RESET} R: {reward}")
        else:
            print(f"T : {targetG} A : {achievedG}{Colors.RESET} R : {reward}")
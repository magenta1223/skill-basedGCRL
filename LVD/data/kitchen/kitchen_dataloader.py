from copy import deepcopy
import numpy as np
from ...collector import Offline_Buffer
# from glob import glob
import gym
from easydict import EasyDict as edict
from ..base_dataset import Base_Dataset
import random
from ...utils import StateProcessor
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH

class Kitchen_Dataset(Base_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        
        # env
        env = gym.make("kitchen-mixed-v0")
        self.dataset = env.get_dataset()
        self.sp = StateProcessor(env_name="kitchen")
        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        # correct terminal points for kitchen 
        for error_idx in [508, 418]:
            seq_end_idxs = np.insert(seq_end_idxs, error_idx, seq_end_idxs[error_idx-1] + 280)

        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len:
                continue    # skip too short demos

            # self.seqs.append(edict(
            #     states=self.dataset['observations'][start:end_idx+1],
            #     actions=self.dataset['actions'][start:end_idx+1],
            # ))


            states=self.dataset['observations'][start:end_idx+1]
            actions=self.dataset['actions'][start:end_idx+1]
            _states = deepcopy(states)

            points = [0]
            unique_subtasks = [""]
            subtasks = []
            goals = []

            n=3
        
            for i, state in enumerate(_states):
                # goal 확인
                g = self.sp.state_goal_checker(state)
                # 달성된게 없으면 다음으로 
                
                if not g:
                    goals.append("".join(unique_subtasks))
                    subtasks.append("")
                    continue      
                goals.append("".join(unique_subtasks))       
                subtask = g[-1]   
                subtasks.append(subtask)


                if len(set(subtasks[-n:] + [subtask])) == 1 and subtask not in unique_subtasks:
                    unique_subtasks.append(subtask)
                    points.append(i) # 


            self.seqs.append(edict(
                states=states,
                actions=actions,
                points = points,
                # subtasks = subtasks,
                # goals = goals,
                # unique_subtasks = unique_subtasks
            ))



            start = end_idx+1

        self.n_seqs = len(self.seqs)
        self.shuffle = self.phase == "train"
        random.shuffle(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.n_repeat = 1
            self.dataset_size = self.val_data_size 
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
            self.n_repeat = 1
            self.dataset_size = self.val_data_size 

        self.count = 0


    def __getitem__(self, index):

        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)

        # G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
        # G[ : self.n_obj] = 0 # only env state

        seg_points = deepcopy(seq.points)
        seg_points = sorted( seg_points + [start_idx])
        start_pos = seg_points.index(start_idx)
        # a가 seg_points의 마지막이라면 -> last state 를 goal로
        # 아니라면 -> 가장 가까운 미래의 seg_points부터 끝까지
        if start_pos == (len(seg_points) - 1):
            g_index = len(seq.states) - 1
        else:
            # print(f"low : {seg_points[start_pos+1]} high : {len(seq.states)}")
            g_index = np.random.randint(low = seg_points[start_pos+1] , high = len(seq.states))

        G = deepcopy(seq.states[g_index])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state



        output = edict(
            states = seq.states[start_idx:start_idx+self.subseq_len, :self.n_obj + self.n_env],
            actions = seq.actions[start_idx:start_idx+self.subseq_len-1],
            G = G
        )

        return output

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)


class Kitchen_Dataset_Div(Kitchen_Dataset):
    """
    """

    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }
    
        self.buffer_prev = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 19, max_size= int(1e5))
        
        # rollout method
        # skill_length = self.subseq_len - 1
        # rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        rollout_length = self.plan_H
        self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= int(1e4))

        self.discount_lambda = np.log(0.99)

    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__

    def enqueue(self, states, actions, c = None):
        self.buffer_now.enqueue(states, actions, c)

    def update_buffer(self):
        self.buffer_prev.copy_from(self.buffer_now)
        # self.buffer_now.reset()

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()
        
    def __skill_learning__(self):
        
        seq = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq.states)

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        # states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.n_obj + self.n_env]
        states = seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]

        # hindsight relabeling 
        # G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
        # G[ : self.n_obj] = 0 # only env state


        seg_points = deepcopy(seq.points)
        seg_points = sorted( seg_points + [start_idx])
        start_pos = seg_points.index(start_idx)
        # a가 seg_points의 마지막이라면 -> last state 를 goal로
        # 아니라면 -> 가장 가까운 미래의 seg_points부터 끝까지
        if start_pos == (len(seg_points) - 1):
            g_index = len(seq.states) - 1
        else:
            # print(f"low : {seg_points[start_pos+1]} high : {len(seq.states)}")
            g_index = np.random.randint(low = seg_points[start_pos+1] , high = len(seq.states))

        G = deepcopy(seq.states[g_index])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state

        output = dict(
            states=states,
            actions=actions,
            G = G,
            rollout = True,
            weights = 1
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio:
            # T, state_dim + action_dim
            # states, actions = self.buffer_prev.sample()
            # states, actions = self.buffer_now.sample()

            # # hindsight relabeling 
            # goal_idx = -1
            # # G = deepcopy(states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]
            # G = deepcopy(states[goal_idx])[:self.n_obj + self.n_env]
            # G[ : self.n_obj] = 0 # only env state
            # output = dict(
            #     states=states[:self.subseq_len],
            #     actions=actions[:self.subseq_len-1],
            #     G=G,
            #     rollout = False,
            #     weights = discount_start * discount_G
            #     # start_idx = 999 #self.novel
            # )



            states, actions, c = self.buffer_now.sample()
            start_idx, goal_idx = self.sample_indices(states)



            # hindsight relabeling
            goal_idx = start_idx + self.subseq_len

            # 새로운 subtask가 등장하면 거기부터 goal로 ㄱㄱ 
            g_achieved = set(self.sp.goal_checker(states[start_idx]))

            while True:
                g = self.sp.goal_checker(states[goal_idx])
                if g:
                    # 새로운 subtask가 존재할 경우 여기부터 goal임. 
                    if set(g) - g_achieved:
                        break
                goal_idx += 1
                if goal_idx == len(states) -1:
                    break
            
            if goal_idx < len(states) - 1:
                goal_idx = np.random.randint(goal_idx, len(states) -1)
            

            # G = deepcopy(states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]
            G = deepcopy(states[goal_idx])[:self.n_obj + self.n_env]
            G[ : self.n_obj] = 0 # only env state

            discount_start = np.exp(self.discount_lambda * (start_idx - c))
            discount_G = np.exp(self.discount_lambda * (goal_idx - c))


            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]

            # trajectory
            output = dict(
                states=states,
                actions=actions,
                G=G,
                rollout = False,
                weights = discount_start * discount_G
                # start_idx = 999 #self.novel
            )

            return output
        else:
            return self.__skill_learning__()
        
class Kitchen_Dataset_Flat(Kitchen_Dataset):

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0])


    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)

        G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state
        
        # 
        output = edict(
            states = seq.states[start_idx, :self.n_obj + self.n_env],
            actions = seq.actions[start_idx],
            G = G
        )

        return output
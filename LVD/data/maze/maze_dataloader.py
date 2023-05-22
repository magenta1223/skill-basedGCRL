from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict

import random
import math
# from proposed.collector.storage import Offline_Buffer
from ...collector.storage import Offline_Buffer

from glob import glob

import h5py
from torch.utils.data import Dataset
from ...contrib.spirl.pytorch_utils import RepeatedDataLoader
import pickle
from torch.utils.data.dataloader import DataLoader, SequentialSampler
import torch
import pickle 

from ..base_dataset import Base_Dataset



def parse_h5(file_path):
    f = h5py.File(file_path)
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
    return edict( 
        states = np.array(f.get("states")),
        actions = np.array(f.get("actions")),
        agent_centric_view = np.array(f.get("images")),
    )

def parse_pkl(file_path):
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


class Maze_Dataset(Base_Dataset):
    # def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)


        with open("./LVD/data/maze/maze_states_skild.pkl", mode ="rb") as f:
            self.seqs = pickle.load(f)

        self.n_seqs = len(self.seqs)
        # self.phase = phase
        # self.dataset_size = dataset_size
        # self.shuffle = shuffle
        # self.device = "cuda"

        # for k, v in data_conf.dataset_spec.items():
        #     setattr(self, k, v) 

        # for k, v in kwargs.items():
        #     setattr(self, k, v)    

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

        self.num = 0

    def __getitem__(self, idx):

        seq = self.seqs[idx]
        states = deepcopy(seq.obs)
        actions = deepcopy(seq.actions)

        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = states[goal_idx][:2]

        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]


        data = {
            'states': states,
            'actions': actions,
            'G' : G,
        }


        return data
    
    def __len__(self):
        return int(self.SPLIT[self.phase] * self.n_seqs)



class Maze_Dataset_Div(Maze_Dataset):
    # def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
    #     super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        self.buffer_dim = self.state_dim

        # 10 step 이후에 skill dynamics로 추론해 error 누적 최소화 
        self.buffer_prev = Offline_Buffer(state_dim= self.buffer_dim, action_dim= self.action_dim, trajectory_length = 19, max_size= 1024)

        
        # rollout method
        skill_length = self.subseq_len - 1
        # 0~11 : 1 skill
        # 12~  : 1skill per timestep
        # total 100 epsiode planning
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
        
        rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        self.buffer_now = Offline_Buffer(state_dim= self.buffer_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)
        # self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)


    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__


    def enqueue(self, states, actions):
        self.buffer_now.enqueue(states, actions)

    def update_buffer(self):
        print("BUFFER RESET!!! ")        
        self.buffer_prev.copy_from(self.buffer_now)
        # self.buffer_now.reset()

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode](index)
        
    def __skill_learning__(self, index):

        seq = self.seqs[index]
        # states = deepcopy(seq['states'])
        states = deepcopy(seq['obs'])
        actions = seq['actions']
        
        # relative position. 
        if self.relative:
            criterion = states[np.random.randint(0, states.shape[0]), :2]
            states[:, :2] -= criterion
        
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"


        


        G = states[goal_idx][:2]
        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]

        data = {
            'states': states,
            'actions': actions,
            'G' : G,
            'rollout' : True
        }

        return data

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            # hindsight relabeling 
            states_images, actions = self.buffer_now.sample()
            
            # # relative position 
            # states_images[:, :2] -= states_images[0, :2]
            # G[ :2] -= states_images[0, :2]

            output = edict(
                states = states_images[:self.subseq_len],
                actions = actions[:self.subseq_len-1],
                G = deepcopy(states_images[-1][:self.n_obj][:2] ),
                rollout = False
                # rollout = True if start_idx < 280 - self.plan_H else False
            )

            return output
        else:
            return self.__skill_learning__(index)

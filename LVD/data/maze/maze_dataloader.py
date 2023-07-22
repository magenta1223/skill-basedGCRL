from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict
from ...collector import Offline_Buffer
from glob import glob
import h5py
import pickle
from ..base_dataset import Base_Dataset
import torch


class Maze_Dataset(Base_Dataset):
    # def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)


        # with open("./LVD/data/maze/maze_states_skild.pkl", mode ="rb") as f:
        #     self.seqs = pickle.load(f)

        with open("./LVD/data/maze/maze_states_skild.pkl", mode ="rb") as f:
            seqs = pickle.load(f)
        
        
        if self.normalize:
            for i in range(len(seqs)):
                seq = seqs[i]
                obs = deepcopy(seq['obs'])
                obs[:, :2] = obs[:, :2] / 40 - 0.5
                obs[:, 2:] = obs[:, 2:] / 10 
                seq['obs'] = obs
            
            self.seqs = seqs 
        else:
            self.seqs = seqs 



        self.n_seqs = len(self.seqs)
        self.shuffle = self.phase == "train"


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

        seq = deepcopy(self.seqs[idx])
        states = deepcopy(seq['obs'])
        actions = deepcopy(seq['actions'])

        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = states[goal_idx][:2]
        # G[2:] = 0

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
        
        # rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        rollout_length = self.plan_H
        self.buffer_now = Offline_Buffer(state_dim= self.buffer_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 10000)
        # self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)


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
        return self.__getitem_methods__[self.mode](index)
        
    def __skill_learning__(self, index):

        seq = deepcopy(self.seqs[index]) # ? 
        # states = deepcopy(seq['states'])
        states = seq['obs']
        actions = seq['actions']
        
        # relative position. 
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = deepcopy(states[goal_idx][:2])
        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]
        
        # if self.normalize:
        #     states[:, :2] = states[:, :2]/40
        #     G[:2] = G[:2]/40

        return edict(
            states = states,
            actions = actions, 
            G = G,
            rollout = True,
            weights = 1
        )

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            # hindsight relabeling 
            states, actions, c = self.buffer_now.sample()
            
            # # relative position 
            # states_images[:, :2] -= states_images[0, :2]
            # G[ :2] -= states_images[0, :2]
            states = states[:self.subseq_len]
            G = deepcopy(states[-1][:self.n_pos][:2])
            G[2:] = 0


            if self.normalize:
                states = torch.clamp(states, 0, 1)
                G = torch.clamp(G, 0, 1)

            return edict(
                states = states,
                actions = actions[:self.subseq_len-1],
                G = G,
                rollout = False,
                weights = 1,
                # rollout = True if start_idx < 280 - self.plan_H else False
            )
        else:
            return self.__skill_learning__(index)

class Maze_Dataset_Div_Sep(Maze_Dataset):
    # def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
    #     super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        
        self.max_generated_seqs = 10000
        self.generated_seqs = []

        self.discount_lambda = np.log(0.99)

    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__


    def enqueue(self, states, actions, c = None, sequence_indices = None):
        for i, seq_idx in enumerate(sequence_indices):
            seq = deepcopy(self.seqs[seq_idx])
            generated_states = states[i]
            generated_actions = actions[i]

            concatenated_states = np.concatenate((seq['obs'][:c[i], :self.state_dim], generated_states), axis = 0)
            concatenated_actions = np.concatenate((seq['actions'][:c[i]], generated_actions), axis = 0)
            
            # 뭔가가 tensor임. 
            new_seq = edict(
                states = concatenated_states,
                actions = concatenated_actions,
                c = c[i].item(),
                seq_index = seq_idx.item()
            )

            self.generated_seqs.append(new_seq)

            if len(self.generated_seqs) > self.max_generated_seqs:
                self.generated_seqs = self.generated_seqs[1:]

    def update_buffer(self):
        pass

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode](index)
        
    def __skill_learning__(self, index):

        seq = deepcopy(self.seqs[index]) # ? 
        # states = deepcopy(seq['states'])
        states = seq['obs']
        actions = seq['actions']
        
        # relative position. 
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = deepcopy(states[goal_idx, :self.n_pos])
        finalG = deepcopy(states[-1, :self.n_pos])

        states = states[start_idx : start_idx + self.subseq_len, :self.state_dim]
        actions = actions[start_idx : start_idx + self.subseq_len -1]
        
        # if self.normalize:
        #     states[:, :2] = states[:, :2]/40
        #     G[:2] = G[:2]/40

        return edict(
            states = states,
            actions = actions, 
            G = G,
            finalG = finalG,
            rollout = True,
            weights = 1,
            seq_index = index,
            start_idx = start_idx
        )

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            # hindsight relabeling 
            _seq_index = np.random.randint(0, len(self.generated_seqs), 1)[0]
            seq = deepcopy(self.generated_seqs[_seq_index])
            states, actions, c, seq_index = seq.states, seq.actions, seq.c, seq.seq_index
            start_idx, goal_idx = self.sample_indices(states)

            goal_idx = -1
            G = deepcopy(states[goal_idx, :self.n_pos])

            # # relative position 
            # states_images[:, :2] -= states_images[0, :2]
            # G[ :2] -= states_images[0, :2]
            # states = states[:self.subseq_len]

            if start_idx > c :
                discount_start = np.exp(self.discount_lambda * (start_idx - c))
            else:
                discount_start = 1
            discount_G = 1

            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]

            return edict(
                states = states,
                actions = actions[:self.subseq_len-1],
                G = G,
                finalG = G,
                rollout = False,
                weights = discount_start * discount_G,
                seq_index = seq_index,
                start_idx = start_idx,

                # rollout = True if start_idx < 280 - self.plan_H else False
            )
        else:
            return self.__skill_learning__(index)


class Maze_Dataset_Flat(Maze_Dataset):
    def __getitem__(self, index):
        seq = self._sample_seq()
        # start_idx = np.random.randint(0, seq.states.shape[0] - 1)
        # goal_idx = -1

        start_idx, goal_idx = self.sample_indices(seq.states)

        states = seq['obs']
        actions = seq['actions']

        G = deepcopy(states[goal_idx, :self.n_pos])
        finalG = deepcopy(states[-1, :self.n_pos])

        states = states[start_idx, :self.state_dim]
        actions = actions[start_idx ]

        # 
        

        output = edict(
            states = seq.states[start_idx, :self.n_pos + self.n_env],
            actions = seq.actions[start_idx],
            G = G
        )

        # print(output.states.shape)
        # print(output.actions.shape)
        # print(output.G.shape)


        return edict(
            states = states,
            actions = actions, 
            G = G,
            finalG = finalG,
            rollout = True,
            weights = 1,
            seq_index = index,
            start_idx = start_idx
        )
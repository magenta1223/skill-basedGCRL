from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict
from glob import glob
import h5py
import pickle
from ..base_dataset import Base_Dataset
import torch


class Maze_Dataset(Base_Dataset):
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)
        with open("./LVD/data/maze/maze_states_skild.pkl", mode ="rb") as f:
            seqs = pickle.load(f)
        
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
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        
        self.buffer_size = cfg.offline_buffer_size 
        
        self.prev_buffer = []
        self.now_buffer = []

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
            
            new_seq = edict(
                states = concatenated_states,
                actions = concatenated_actions,
                c = c[i].item(),
                seq_index = seq_idx.item()
            )

            self.prev_buffer.append(new_seq)

            if len(self.prev_buffer) > self.buffer_size:
                self.prev_buffer = self.prev_buffer[1:]

    def update_buffer(self):
        self.now_buffer = deepcopy(self.prev_buffer)
        self.prev_buffer = []
        pass
    

    def __getitem__(self, index):
        return self.__getitem_methods__[self.mode](index)
        
    def __skill_learning__(self, index):

        seq = deepcopy(self.seqs[index]) 
        states = seq['obs']
        actions = seq['actions']
        
        # relative position. 
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = deepcopy(states[goal_idx, :self.n_pos])
        finalG = deepcopy(states[-1, :self.n_pos])

        states = states[start_idx : start_idx + self.subseq_len, :self.state_dim]
        actions = actions[start_idx : start_idx + self.subseq_len -1]
        
        return edict(
            states = states,
            actions = actions, 
            G = G,
            finalG = finalG,
            rollout = True,
            weights = 1,
            seq_index = index,
            start_idx = start_idx,
            seq_len = len(seq['obs'])
        )

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            # hindsight relabeling 
            _seq_index = np.random.randint(0, len(self.now_buffer), 1)[0]
            seq = deepcopy(self.now_buffer[_seq_index])
            states, actions, c, seq_index = seq.states, seq.actions, seq.c, seq.seq_index
            start_idx, goal_idx = self.sample_indices(states)

            G = deepcopy(states[goal_idx, :self.n_pos])

            discount_start = np.exp(self.discount_lambda *  max((start_idx - c), 0))
            discount_G = np.exp(self.discount_lambda *  max((goal_idx - c), 0))

            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]

            return edict(
                states = states,
                actions = actions[:self.subseq_len-1],
                G = G,
                finalG = G,
                rollout = False,
                weights = discount_start * discount_G if self.discount else 1,
                seq_index = seq_index,
                start_idx = start_idx,
                seq_len = 1,
            )
        else:
            return self.__skill_learning__(index)


class Maze_Dataset_Flat(Maze_Dataset):
    def __getitem__(self, index):
        seq = self._sample_seq()
        states = seq['obs']
        actions = seq['actions']

        start_idx, goal_idx = self.sample_indices(states)

        G = deepcopy(states[goal_idx, :self.n_pos])
        finalG = deepcopy(states[-1, :self.n_pos])

        states = states[start_idx, :self.state_dim]
        actions = actions[start_idx ]

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

    def __len__(self):
        return super().__len__() * 10 


class Maze_Dataset_Flat_WGCSL(Maze_Dataset):

    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        self.discount = np.log(0.99)
    
    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """
        goal_max_index = len(states) - 1 
        start_idx = np.random.randint(min_idx, states.shape[0] - 1)
        goal_index = np.random.randint(start_idx, goal_max_index)
        return start_idx, goal_index


    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq['obs'])
        G = deepcopy(seq['obs'][goal_idx, :self.n_pos])
        states = seq['obs'][start_idx, :self.state_dim]
        next_states = seq['obs'][start_idx + 1, :self.state_dim]
        actions = seq['actions'][start_idx]
        drw = np.exp(self.discount * (goal_idx - start_idx))

        if self.distance_reward:
            if np.linalg.norm(states[:self.n_pos]- G) < 1:
                reward, done = 1, 1
            else:
                reward, done = 0, 0

        else:
            if goal_idx - start_idx < self.reward_threshold:
                reward, done = 1, 1
            else:
                reward, done = 0, 0 

        return edict(
            states = states,
            actions = actions, 
            next_states = next_states,
            G = G,
            reward = reward,
            done = done,
            drw = drw
        )
    
    def __len__(self):
        return super().__len__() * 10 
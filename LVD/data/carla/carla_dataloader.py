from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict

import random
import math
# from proposed.collector.storage import Offline_Buffer
from ...collector import Offline_Buffer

from glob import glob
import pickle 
from ..base_dataset import Base_Dataset

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class Carla_Dataset(Base_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)

        with open(f"./LVD/data/carla/carla_data.pkl", mode ="rb") as f:
            self.seqs = pickle.load(f)        

        
        # with open(f"./LVD/data/carla/carla_dataset{mode}.pkl", mode ="rb") as f:
        #     self.seqs = pickle.load(f)

        

        self.n_seqs = len(self.seqs)
        self.phase = phase
        self.shuffle = self.phase == "train"
        self.device = "cuda"

        nObs = sum([seq.actions.shape[0] for seq in self.seqs])
            


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


        self.num = 0
        self.nObs = nObs
    
    @staticmethod 
    def parseObs(obs):
        return obs['position'], np.concatenate(list(obs.values()), axis = -1) 


    def __getitem__(self, index):

        seq = self._sample_seq()
        positions, states = self.parseObs(seq['obs'])
        actions = seq['actions']

        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = positions[goal_idx]

        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]


        data = edict(
            states = states,
            actions = actions,
            G = G,
        )

        return data
    
    def __len__(self):
        return  int(self.SPLIT[self.phase] * self.nObs)


class Carla_Dataset_Div(Carla_Dataset):
    def __init__(self, cfg, phase):
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
            
            # start idx도 필요
            concatenated_states = np.concatenate((seq.states[:c[i], :self.state_dim], generated_states), axis = 0)
            concatenated_actions = np.concatenate((seq.actions[:c[i]], generated_actions), axis = 0)
            
            # np.savez("./unseen_G_states.npz", states = concatenated_states, actions = concatenated_actions)
            # sequnece가 원래것과 매칭이 안되고 있음. 버그
            # assert 1==0, "a"
            

            # g = \varphi(s) 인 \varphi는 알고 있음을 가정. (뭐가 열렸는지 돌아갔는지 정도는 알 수 있음.)
            # 그러면 유의한 goal state가 뭔지 정도는 알 수 있음. 
            # = 유의미한 goal 변화 없으면 거기서 컽
            new_seq = edict(
                states = concatenated_states,
                actions = concatenated_actions,
                c = c[i].item(),
                seq_index = seq_idx.item()
            )

            self.generated_seqs.append(new_seq)

            if len(self.generated_seqs) > self.max_generated_seqs:
                self.generated_seqs = self.generated_seqs[1:]

        # np.savez("./unseen_G_states.npz", states = concatenated_states, actions = concatenated_actions)

    def update_buffer(self):
        pass 

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()
        
    def __skill_learning__(self):
        seq, index = self._sample_seq(return_index= True)

        positions, states = self.parseObs(seq['obs'])
        actions = seq['actions']
        
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = positions[goal_idx].astype(np.float32)
        states = states[start_idx : start_idx + self.subseq_len].astype(np.float32)
        actions = actions[start_idx : start_idx + self.subseq_len -1].astype(np.float32)

        output = edict(
            states=states,
            actions=actions,
            G = G,
            rollout = True,
            weights = 1,
            seq_index = index,
            start_idx = start_idx,
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio:
            _seq_index = np.random.randint(0, len(self.generated_seqs), 1)[0]
            seq = deepcopy(self.generated_seqs[_seq_index])


            states, actions, c, seq_index = seq.states, seq.actions, seq.c, seq.seq_index
            start_idx, goal_idx = self.sample_indices(states)

            goal_idx = -1

            G = states[goal_idx][12:14]
            if start_idx > c :
                discount_start = np.exp(self.discount_lambda * (start_idx - c))
            else:
                discount_start = 1
            # discount_G = np.exp(self.discount_lambda * (goal_idx - c))
            discount_G = 1


            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]
            output = edict(
                states=states,
                actions=actions,
                G=G,
                rollout = False,
                weights = 1, #discount_start * discount_G,
                seq_index = seq_index,
                start_idx = start_idx
                # start_idx = 999 #self.novel
            )

            return output
        else:
            return self.__skill_learning__()

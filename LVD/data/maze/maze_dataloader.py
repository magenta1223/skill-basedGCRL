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
            
            # 뭔가가 tensor임. 
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

            goal_idx = -1
            G = deepcopy(states[goal_idx, :self.n_pos])

            # # relative position 
            # states_images[:, :2] -= states_images[0, :2]
            # G[ :2] -= states_images[0, :2]
            # states = states[:self.subseq_len]

            # if start_idx > c :
            #     discount_start = np.exp(self.discount *  max((start_idx - c), 0))
            # else:
            #     discount_start = 1

            # # discount_G = np.exp(self.discount_lambda * (goal_idx - c))
            # discount_G = 1
            # if goal_idx > c :
            #     discount_G = np.exp(self.discount *  max((goal_idx - c), 0))
            # else:
            #     discount_G = 1

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

                # rollout = True if start_idx < 280 - self.plan_H else False
            )
        else:
            return self.__skill_learning__(index)


class Maze_Dataset_Flat(Maze_Dataset):
    def __getitem__(self, index):
        seq = self._sample_seq()
        # start_idx = np.random.randint(0, seq.states.shape[0] - 1)
        # goal_idx = -1

        states = seq['obs']
        actions = seq['actions']

        start_idx, goal_idx = self.sample_indices(states)



        G = deepcopy(states[goal_idx, :self.n_pos])
        finalG = deepcopy(states[-1, :self.n_pos])

        states = states[start_idx, :self.state_dim]
        actions = actions[start_idx ]

        # 
        

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


    def __len__(self):
        return super().__len__() * 10 # skill horizon  # len traj



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
        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
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

        # if goal_idx - start_idx < self.reward_threshold:
        #     reward, done = 1, 1
        # else:
        #     reward, done = -1, 0 

        if np.linalg.norm(states[:self.n_pos]- G) < 1:
            reward, done = 1, 1
        else:
            reward, done = 0, 0


        # print(output.states.shape)
        # print(output.actions.shape)
        # print(output.G.shape)

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
        return super().__len__() * 10 # skill horizon  # len traj




class Maze_Dataset_Flat_RIS(Maze_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        self.discount = np.log(0.99)
        
        self.all_states = np.concatenate([seq['obs'] for seq in self.seqs], axis = 0)
            
    


    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """

        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
        # start_idx = np.random.randint(min_idx, states.shape[0] - 1)
        # goal_index = np.random.randint(start_idx, goal_max_index)

        indices = np.random.randint(min_idx, goal_max_index, 2)
        start_idx, goal_index = indices.min(), indices.max()


        return start_idx, goal_index


    def __getitem__(self, index):
        # seq = self._sample_seq()
        
        seq_index = index % len(self.seqs)

        print(seq_index)

        seq = self.seqs[seq_index]

        start_idx, goal_idx = self.sample_indices(seq['obs'])
        # subgoal_index = np.random.randint(start_idx, goal_idx)
        subgoal_index = np.random.randint(0, len(self.all_states) - 1)
        G = deepcopy(seq['obs'][goal_idx])[:self.n_pos]
        G[ : self.n_pos] = 0 # only env state
        reward = 1 if goal_idx - start_idx < self.reward_threshold else 0
        
        # discounted relabeling weight 
        # drw = np.exp(self.discount * (goal_idx - start_idx))
        output = edict(
            states = seq['obs'][start_idx, :self.state_dim],
            actions = seq['actions'][start_idx],
            subgoals = self.all_states[subgoal_index, :self.state_dim],
            next_states = seq['obs'][start_idx + 1, :self.state_dim],
            G = G,
            reward = reward,
            # drw = drw
            done = 1 if reward else 0
        )

        return output
    
    def __len__(self):
        return super().__len__() * 10 # skill horizon  # len traj



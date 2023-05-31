from copy import deepcopy
import numpy as np
from ...collector import Offline_Buffer
# from glob import glob
import gym
from easydict import EasyDict as edict
from ..base_dataset import Base_Dataset

class Kitchen_Dataset(Base_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        
        # env
        env = gym.make("kitchen-mixed-v0")
        self.dataset = env.get_dataset()
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

            self.seqs.append(edict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))

            start = end_idx+1

        self.n_seqs = len(self.seqs)

        self.shuffle = self.phase == "train"

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

        G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
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
        skill_length = self.subseq_len - 1
        rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)


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
        G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state

        output = dict(
            states=states,
            actions=actions,
            G = G,
            rollout = True
            # rollout = True if start_idx < 280 - self.plan_H else False
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio:
            # T, state_dim + action_dim
            # states, actions = self.buffer_prev.sample()
            states, actions = self.buffer_now.sample()



            # hindsight relabeling 
            goal_idx = -1
            # G = deepcopy(states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]
            G = deepcopy(states[goal_idx])[:self.n_obj + self.n_env]
            G[ : self.n_obj] = 0 # only env state

            # trajectory
            output = dict(
                states=states[:self.subseq_len],
                actions=actions[:self.subseq_len-1],
                G=G,
                rollout = False,
                # start_idx = 999 #self.novel
            )

            return output
        else:
            return self.__skill_learning__()
        
class Kitchen_Dataset_Flat(Kitchen_Dataset):

    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)

        G = deepcopy(seq.states[-1])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state

        output = edict(
            states = seq.states[start_idx, :self.n_obj + self.n_env],
            actions = seq.actions[start_idx],
            G = G
        )

        return output
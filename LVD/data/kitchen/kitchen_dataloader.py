from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
# from proposed.collector.storage import Offline_Buffer
from ...collector.storage import Offline_Buffer
from ...contrib.spirl.pytorch_utils import RepeatedDataLoader
from glob import glob
import gym

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3




class D4RLSequenceSplitDataset(Dataset):
    SPLIT = dict(train=0.99, val=0.01, test=0.0)

    def __init__(self, cfg, phase):
        
        for k, v in cfg.items():
            setattr(self, k , v)

        self.phase = phase

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

            self.seqs.append(dict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))

            start = end_idx+1

        self.n_seqs = len(self.seqs)

        self.shuffle = self.phase == "train"

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT['train'] * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT['train'] * self.n_seqs)
            self.end = int((self.SPLIT['train'] + self.SPLIT['val']) * self.n_seqs)
            self.n_repeat = 1
            self.dataset_size = self.val_data_size 
        else:
            self.start = int((self.SPLIT['train'] + self.SPLIT['val']) * self.n_seqs)
            self.end = self.n_seqs
            self.n_repeat = 1
            self.dataset_size = self.val_data_size 


    def __getitem__(self, index):
        return NotImplementedError

    def _sample_seq(self, index = False):
        try:
            if index:
                print(f"index {index}")
                return self.seqs[index]
            else:
                return np.random.choice(self.seqs[self.start:self.end])
        except:
            return self.seqs[-1]

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)

    def get_data_loader(self, batch_size, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))

        return RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            drop_last=False,
            n_repeat=self.epoch_cycles_train,
            pin_memory= True,
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
            )


class Kitchen_Dataset(D4RLSequenceSplitDataset):
    def __init__(self, cfg, phase = "train"):
        super().__init__(cfg, phase)

    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """

        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
        start_idx = np.random.randint(min_idx, states.shape[0] - self.subseq_len - 1)
        
        goal_index = np.random.randint(start_idx + self.subseq_len, goal_max_index)

        return start_idx, goal_index


    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq['states'])

        G = deepcopy(seq['states'][goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state

        output = dict(
            states = seq['states'][start_idx:start_idx+self.subseq_len, :self.n_obj + self.n_env],
            actions = seq['actions'][start_idx:start_idx+self.subseq_len-1],
            G = G
        )

        return output


class Kitchen_Dataset_Div(Kitchen_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.__mode__ = "skill_learning"
        
        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        self.buffer_prev = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 50, max_size= 100000)
        self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 50, max_size= 100000)

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
        self.buffer_now.reset()

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()


    def __skill_learning__(self):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq['states'].shape[0] - self.subseq_len - 1)
        output = dict(
            states = seq['states'][start_idx:start_idx+self.subseq_len, :self.n_obj + self.n_env],
            actions = seq['actions'][start_idx:start_idx+self.subseq_len-1],
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() > 0.95:
            states, actions = self.buffer_prev.sample()

            # trajectory
            output = dict(
                states=states[:self.subseq_len],
                actions=actions[:self.subseq_len-1],
            )

            return output
        else:
            return self.__skill_learning__()


class Kitchen_Dataset_GC(Kitchen_Dataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """


    def __getitem__(self, index):
        # sample start index in data range
        seq = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq.states)
        
        # trajectory
        states = seq['states'][start_idx : start_idx+self.subseq_len][:, :self.n_obj + self.n_env]
        actions = seq['actions'][start_idx:start_idx+self.subseq_len-1]
        
        # goal final
        G = deepcopy(seq['states'][goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state
        # G[ self.n_obj + self.n_env : ] = 0  # goal state 밀기 어차피 분리해서 받을거 .


        output = dict(
            # relabeled_inputs = relabeled_inputs,
            states=states,
            actions=actions,
            G = G,
        )

        return output


class Kitchen_Dataset_GC_Div(Kitchen_Dataset_GC):
    """
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }
    
        # print("STATE DIM", self.state_dim)
        
        # 10 step 이후에 skill dynamics로 추론해 error 누적 최소화 
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
        
        seq_skill = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq_skill['states'])

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        # states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.n_obj + self.n_env]
        states = seq_skill['states'][start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq_skill['actions'][start_idx:start_idx+self.subseq_len-1]

        # hindsight relabeling 
        G = deepcopy(seq_skill['states'][goal_idx])[:self.n_obj + self.n_env]
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

            if self.only_proprioceptive:
                states = states[:, :self.n_obj]

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
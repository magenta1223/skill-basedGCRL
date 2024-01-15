from copy import deepcopy
import numpy as np
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
                start = end_idx -1
                continue    # skip too short demos

            states=self.dataset['observations'][start:end_idx+1]
            actions=self.dataset['actions'][start:end_idx+1]
            _states = deepcopy(states)

            points = [0]
            unique_subtasks = [""]
            subtasks = []
            goals = []
            
            n=3
            for i, state in enumerate(_states):
                g = self.sp.state_goal_checker(state)
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

    def sample_indices(self, states, min_idx=0):
        """
        \Phi(s_t) != \Phi(s_i)
        """
        goal_max_index = len(states) - 1 # 
        start_idx = np.random.randint(min_idx, states.shape[0] - self.subseq_len - 1)
        
        g_min_index = start_idx
        while True:
            goal_index = np.random.randint(g_min_index + self.subseq_len, goal_max_index)
            
            # 비교 
            
            phi_st = deepcopy(states[start_idx])[:self.state_dim]
            phi_st[ : self.n_pos] = 0 # only env state
            
            phi_si = deepcopy(states[goal_index])[:self.state_dim]
            phi_si[ : self.n_pos] = 0 # only env state
            
            if np.linalg.norm(phi_st - phi_si) < 0.2:
                if g_min_index + 10 > goal_max_index - self.subseq_len - 1:
                    goal_index = goal_max_index
                    break 
                g_min_index = min(goal_max_index - self.subseq_len - 1, g_min_index + 10)
            else:
                break

        return start_idx, goal_index


    def __getitem__(self, index):

        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)

        # assert start_idx < goal_idx, "Invalid"

        states = deepcopy(seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim])
        actions = deepcopy(seq.actions[start_idx:start_idx+self.subseq_len-1])
        
        if self.use_sp:
            seg_points = deepcopy(seq.points)
            seg_points = sorted( seg_points + [start_idx])
            start_pos = seg_points.index(start_idx)
            if start_pos == (len(seg_points) - 1):
                goal_idx = len(seq.states) - 1
            else:
                goal_idx = np.random.randint(low = seg_points[start_pos+1] , high = len(seq.states))

        G = deepcopy(seq.states[goal_idx])[:self.state_dim]
        G[ : self.n_pos] = 0 # only env state

        output = edict(
            states = states,
            actions = actions,
            G = G
        )

        return output

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)
    

class Kitchen_Dataset_Div_Sep(Kitchen_Dataset):
    pass 



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
        
        self.buffer_size = cfg.offline_buffer_size 
        self.prev_buffer = []
        self.now_buffer = []

        # self.discount_lambda = np.log(0.99)
        
        discount = cfg.disc_pretrain
        self.do_discount = discount.apply
        self.static_discount = discount.static
        self.discount_raw = discount.start
        # self.discount_lambda = np.log(discount.start)
        self.max_discount = discount.end
        self.discount_interval = (discount.end - discount.start) / (discount.epochs - self.mixin_start)
        # self.discount_lambda = np.log(cfg.discount_value)
        
        
        
        mixin_ratio = self.mixin_ratio 
        self.static_ratio = mixin_ratio.static
        self.mixin_ratio = mixin_ratio.start         
        self.max_mixin_ratio = mixin_ratio.end
        self.ratio_interval = (mixin_ratio.end - mixin_ratio.start) / (mixin_ratio.epochs - self.mixin_start)

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
            
            concatenated_states = np.concatenate((seq.states[:c[i], :self.state_dim], generated_states), axis = 0)[:280]
            concatenated_actions = np.concatenate((seq.actions[:c[i]], generated_actions), axis = 0)[:280]
            
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
        
    def update_ratio(self):        
        if not self.static_ratio:
            self.mixin_ratio += self.ratio_interval
            if self.ratio_interval > 0:
                cutoff_func = min
            else:
                cutoff_func = max 
            self.mixin_ratio = cutoff_func(self.mixin_ratio, self.max_mixin_ratio)    
        
        if not self.static_discount:
            self.discount_raw += self.discount_interval
            if self.discount_interval > 0:
                cutoff_func = min
            else:
                cutoff_func = max
            self.discount_raw = cutoff_func(self.discount_raw, self.max_discount)
            
    def __getitem__(self, index):
        return self.__getitem_methods__[self.mode]()
    
        
    def __skill_learning__(self):
        
        seq, index = self._sample_seq(return_index= True)
        start_idx, goal_idx = self.sample_indices(seq.states)

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        states = seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]

        # hindsight relabeling 

        if self.use_sp:
            seg_points = deepcopy(seq.points)
            seg_points = sorted( seg_points + [start_idx])
            start_pos = seg_points.index(start_idx)
            if start_pos == (len(seg_points) - 1):
                goal_idx = len(seq.states) - 1
            else:
                goal_idx = np.random.randint(low = seg_points[start_pos+1] , high = len(seq.states))


        G = deepcopy(seq.states[goal_idx])[:self.state_dim]
        G[ : self.n_pos] = 0 # only env state

        output = edict(
            states=states,
            actions=actions,
            G = G,
            rollout = True,
            weights = 1,
            seq_index = index,
            start_idx = start_idx,
            seq_len = len(seq.states)
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio and len(self.now_buffer) > 0:
            _seq_index = np.random.randint(0, len(self.now_buffer), 1)[0]
            seq = deepcopy(self.now_buffer[_seq_index])
            states, actions, c, seq_index = seq.states, seq.actions, seq.c, seq.seq_index
            start_idx, goal_idx = self.sample_indices(states)
            
            # remove? 
            # goal_idx = -1
            G = deepcopy(states[goal_idx])[:self.state_dim]
            G[ : self.n_pos] = 0 # only env state
            
            discount_start = np.exp(self.discount_lambda *  max((start_idx - c), 0))
            discount_G = np.exp(self.discount_lambda *  max((goal_idx - c), 0))

            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]

            # trajectory
            output = edict(
                states=states,
                actions=actions,
                G=G,
                rollout = False,
                weights = discount_start * discount_G if self.discount else 1,
                seq_index = seq_index,
                start_idx = start_idx,
                seq_len = len(seq.states)                
            )

            return output
        else:
            return self.__skill_learning__()
        
        
    def update_ratio(self):        
        self.mixin_ratio += self.ratio_interval
        self.mixin_ratio = min(self.mixin_ratio, self.max_mixin_ratio)    
    @property
    def discount_lambda(self):
        return np.log(self.discount_raw)
        
        
class Kitchen_Dataset_Flat(Kitchen_Dataset):
    def __len__(self):
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0])

    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - 1)
        goal_idx = -1

        G = deepcopy(seq.states[goal_idx])[:self.n_pos + self.n_env]
        G[ : self.n_pos] = 0 # only env state
        
        
        
        
        output = edict(
            states = seq.states[start_idx, :self.n_pos + self.n_env],
            actions = seq.actions[start_idx],
            G = G
        )
        
        
        return output
        
class Kitchen_Dataset_Flat_WGCSL(Kitchen_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        self.discount = np.log(0.99)
    
    def __len__(self):
        # if self.dataset_size != -1:
        #     return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0])

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
        start_idx, goal_idx = self.sample_indices(seq.states)

        G = deepcopy(seq.states[goal_idx])[:self.n_pos + self.n_env]
        G[ : self.n_pos] = 0 # only env state

        if goal_idx - start_idx < self.reward_threshold:
            reward, done = 1, 1
        else:
            reward, done = 0, 0 
        
        # discounted relabeling weight 
        drw = np.exp(self.discount * (goal_idx - start_idx))
        
        output = edict(
            states = seq.states[start_idx, :self.n_pos + self.n_env],
            actions = seq.actions[start_idx],
            next_states = seq.states[start_idx + 1, :self.n_pos + self.n_env],
            G = G,
            reward = reward,
            done = done,
            drw = drw
        )
        return output
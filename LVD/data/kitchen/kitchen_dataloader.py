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

        # assert start_idx < goal_idx, "Invalid"

        states = deepcopy(seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim])
        actions = deepcopy(seq.actions[start_idx:start_idx+self.subseq_len-1])

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

        G = deepcopy(seq.states[g_index])[:self.state_dim]
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
        
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        states = seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]

        # hindsight relabeling 

        seg_points = seq.points
        seg_points = sorted( seg_points + [start_idx])
        start_pos = seg_points.index(start_idx)
        # a가 seg_points의 마지막이라면 -> last state 를 goal로
        # 아니라면 -> 가장 가까운 미래의 seg_points부터 끝까지
        if start_pos == (len(seg_points) - 1):
            g_index = len(seq.states) - 1
        else:
            # print(f"low : {seg_points[start_pos+1]} high : {len(seq.states)}")
            g_index = np.random.randint(low = seg_points[start_pos+1] , high = len(seq.states))

        G = deepcopy(seq.states[g_index])[:self.n_pos + self.n_env]
        G[ : self.n_pos] = 0 # only env state

        output = dict(
            states=states,
            actions=actions,
            G = G,
            rollout = True,
            weights = 1
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio and self.buffer_now.size > 0:

            states, actions, c = self.buffer_now.sample()
            start_idx, goal_idx = self.sample_indices(states)

            # # hindsight relabeling
            # goal_idx = start_idx + self.subseq_len

            # # 새로운 subtask가 등장하면 거기부터 goal로 ㄱㄱ 
            # g_achieved = set(self.sp.goal_checker(states[start_idx]))

            # while True:
            #     g = self.sp.goal_checker(states[goal_idx])
            #     if g:
            #         # 새로운 subtask가 존재할 경우 여기부터 goal임. 
            #         if set(g) - g_achieved:
            #             break
            #     goal_idx += 1
            #     if goal_idx == len(states) -1:
            #         break
            
            # if goal_idx < len(states) - 1:
            #     goal_idx = np.random.randint(goal_idx, len(states) -1)
            
            goal_idx = -1
            G = deepcopy(states[goal_idx])[:self.n_pos + self.n_env]
            G[ : self.n_pos] = 0 # only env state

            discount_start = np.exp(self.discount_lambda * max((start_idx - c), 0))
            # discount_G = np.exp(self.discount_lambda * (goal_idx - c))
            discount_G = 1


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
        # if self.dataset_size != -1:
        #     return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0])


    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - 1)
        goal_idx = -1


        # start_idx, goal_idx = self.sample_indices(seq.states)

        G = deepcopy(seq.states[goal_idx])[:self.n_pos + self.n_env]
        G[ : self.n_pos] = 0 # only env state
        
        # 
        

        output = edict(
            states = seq.states[start_idx, :self.n_pos + self.n_env],
            actions = seq.actions[start_idx],
            G = G
        )

        # print(output.states.shape)
        # print(output.actions.shape)
        # print(output.G.shape)


        return output
    


class Kitchen_Dataset_Div_Sep(Kitchen_Dataset):
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

        self.discount = np.log(0.99)

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
            concatenated_states = np.concatenate((seq.states[:c[i], :self.state_dim], generated_states), axis = 0)[:280]
            concatenated_actions = np.concatenate((seq.actions[:c[i]], generated_actions), axis = 0)[:280]
            
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

            self.prev_buffer.append(new_seq)

            if len(self.prev_buffer) > self.buffer_size:
                self.prev_buffer = self.prev_buffer[1:]

    def update_buffer(self):
        self.now_buffer = deepcopy(self.prev_buffer)
        self.prev_buffer = []
        pass

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()
        
    def __skill_learning__(self):
        
        seq, index = self._sample_seq(return_index= True)
        start_idx, goal_idx = self.sample_indices(seq.states)

        # assert start_idx < goal_idx, "Invalid"

        # trajectory
        states = seq.states[start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]

        # hindsight relabeling 


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

        G = deepcopy(seq.states[g_index])[:self.state_dim]
        G[ : self.n_pos] = 0 # only env state

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

        if np.random.rand() < self.mixin_ratio and len(self.now_buffer) > 0:
            _seq_index = np.random.randint(0, len(self.now_buffer), 1)[0]
            seq = deepcopy(self.now_buffer[_seq_index])
            states, actions, c, seq_index = seq.states, seq.actions, seq.c, seq.seq_index
            start_idx, goal_idx = self.sample_indices(states)
            
            goal_idx = -1
            G = deepcopy(states[goal_idx])[:self.state_dim]
            G[ : self.n_pos] = 0 # only env state
            
            # 만약 start_idx가 c보다 나중이면 -> discount..해야겠지? 
            if start_idx > c :
                discount_start = np.exp(self.discount *  max((start_idx - c), 0))
            else:
                discount_start = 1

            # discount_G = np.exp(self.discount_lambda * (goal_idx - c))
            discount_G = 1
            if goal_idx > c :
                discount_G = np.exp(self.discount *  max((goal_idx - c), 0))

            
            else:
                discount_G = 1


            states = states[start_idx : start_idx+self.subseq_len, :self.state_dim]
            actions = actions[start_idx:start_idx+self.subseq_len-1]

            # trajectory
            output = edict(
                states=states,
                actions=actions,
                G=G,
                rollout = False,
                weights = discount_start * discount_G,
                seq_index = seq_index,
                start_idx = start_idx
                # start_idx = 999 #self.novel
            )

            return output
        else:
            return self.__skill_learning__()
        
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

        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
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
            reward, done = -1, 0 
        
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
    
class Kitchen_Dataset_Flat_RIS(Kitchen_Dataset):
    def __init__(self, cfg, phase):
        super().__init__(cfg, phase)
        self.discount = np.log(0.99)
        
        self.all_states = np.concatenate([seq.states for seq in self.seqs], axis = 0)
            
    
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

        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
        start_idx = np.random.randint(min_idx, states.shape[0] - 1)
        
        goal_index = np.random.randint(start_idx, goal_max_index)

        return start_idx, goal_index


    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)


        # subgoal_index = np.random.randint(start_idx, goal_idx)
        
        subgoal_index = np.random.randint(0, len(self.all_states) - 1)

        G = deepcopy(seq.states[goal_idx])[:self.n_pos + self.n_env]
        G[ : self.n_pos] = 0 # only env state
        reward =  1 if goal_idx - start_idx < self.reward_threshold else 0 
        
        # discounted relabeling weight 
        # drw = np.exp(self.discount * (goal_idx - start_idx))

    
        output = edict(
            states = seq.states[start_idx, :self.n_pos + self.n_env],
            actions = seq.actions[start_idx],
            subgoals = self.all_states[subgoal_index, :self.n_pos + self.n_env],
            next_states = seq.states[start_idx + 1, :self.n_pos + self.n_env],
            G = G,
            reward = reward,
            # drw = drw
            done = 1 if reward else 0
        )

        return output
import numpy as np
from torch.utils.data import Dataset
from ..contrib.spirl.pytorch_utils import RepeatedDataLoader 
from easydict import EasyDict as edict
import random
from copy import deepcopy

class Base_Dataset(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)

    def __init__(self, cfg, phase):
        
        for k, v in cfg.items():
            setattr(self, k , v)

        self.phase = phase
        self.seqs = []
        self.start = 0
        self.end = -1


    def _sample_seq(self, index= False, return_index= False):
        try:
            if index:
                print(f"index {index}")
                if return_index:
                    return deepcopy(self.seqs[index]), index
                else:
                    return deepcopy(self.seqs[index])
            else:
                # return np.random.choice(self.seqs[self.start:self.end])
                random_index = random.randint(self.start, self.end - 1)
                if return_index:
                    return deepcopy(self.seqs[random_index]), random_index
                else:
                    return deepcopy(self.seqs[random_index])

        except:
            if return_index:
                return deepcopy(self.seqs[-1]), len(self.seqs) - 1
            else:
                return deepcopy(self.seqs[-1])


    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """

        goal_max_index = len(states) - 1 # 
        start_idx = np.random.randint(min_idx, states.shape[0] - self.subseq_len - 1)

        goal_index = np.random.randint(start_idx + self.subseq_len, goal_max_index)

        return start_idx, goal_index


    def __len__(self):
        return NotImplementedError

    def get_data_loader(self, batch_size, num_workers):
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

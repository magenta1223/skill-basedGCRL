from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
# from .general_utils import batchwise_assign
import numpy as np

def batchwise_assign(tensor, index, value):
    """ Assigns the _tensor_ elements at the _index_ the _value_. The indexing is along dimension 1

    :param tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 where that t2[i, index[i]] = value
    """
    bs = tensor.shape[0]
    tensor[np.arange(bs), index] = value


class RepeatedDataLoader(DataLoader):
    """ A data loader that returns an iterator cycling through data n times """
    def __init__(self, *args, n_repeat=1, **kwargs):
        # print(args, kwargs)
        # (<LVD.data.kitchen.kitchen_dataloader.Kitchen_Dataset_Div object at 0x7fe025c79af0>,) {'batch_size': 1024, 'shuffle': True, 'num_workers': 16, 'drop_last': False, 'pin_memory': True, 'worker_init_fn': <function Base_Dataset.get_data_loader.<locals>.<lambda> at 0x7fdf1e8d6f70>} 
        # (<LVD.data.ant.antmaze_dataloader.Antmaze_Dataset_Div object at 0x7fc332b7d7c0>,) {'batch_size': 1024, 'shuffle': True, 'num_workers': 16, 'drop_last': False, 'pin_memory': True, 'worker_init_fn': <function Base_Dataset.get_data_loader.<locals>.<lambda> at 0x7fc332b045e0>}
        
        super().__init__(*args, **kwargs)
        if n_repeat != 1:
            self._DataLoader__initialized = False   # this is an ugly hack for pytorch1.3 to be able to change the attr
            self.batch_sampler = RepeatedSampler(self.batch_sampler, n_repeat)
            self._DataLoader__initialized = True

    # additional methods
    def set_sampler(self, sampler):
        self.batch_sampler = sampler

    def set_mode(self, mode):
        self.dataset.set_mode(mode)
    
    def enqueue(self, states, actions, *args, **kwargs):
        self.dataset.enqueue(states, actions, *args, **kwargs)

    def update_buffer(self):
        self.dataset.update_buffer()
    
    def update_ratio(self):
        self.dataset.update_ratio()

class RepeatedSampler(Sampler):
    """ A sampler that repeats the data n times """
    
    def __init__(self, sampler, n_repeat):
        super().__init__(sampler)
        
        self._sampler = sampler
        self.n_repeat = n_repeat
        
    def __iter__(self):
        for i in range(self.n_repeat):
            for elem in self._sampler:
                yield elem

    def __len__(self):
        return len(self._sampler) * self.n_repeat
    
    
def like(func, tensor):
    return partial(func, device=tensor.device, dtype=tensor.dtype)


def make_one_hot(index, length):
    """ Converts indices to one-hot values"""
    oh = index.new_zeros([index.shape[0], length])
    batchwise_assign(oh, index, 1)
    return oh
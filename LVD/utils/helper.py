import os
import random
import time
from datetime import datetime

import torch
from torch.nn import functional as F
import torch.distributions as torch_dist
from torch.distributions.kl import register_kl

import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES, BONUS_THRESH



from ..contrib.dists import TanhNormal

# --------------------------- Seed --------------------------- #

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    # torch.use_deterministic_algorithms(True)


# --------------------------- Env, Model Utils --------------------------- #

def prep_state(states, device):
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype = torch.float32)

    if len(states.shape) < 2:
        states = states.unsqueeze(0)

    states = states.to(device)
    return states


# --------------------- Helper Class --------------------- # 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Scheduler_Helper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, module_name = ""):

        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)
        self.module_name = module_name

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of {}`s group {} to {:.4e}.'.format(epoch_str, self.module_name, i, new_lr))
                    



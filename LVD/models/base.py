import torch
import torch.nn as nn
import torch.distributions as torch_dist
from ..modules import *
from ..utils import *
# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 


class BaseModel(BaseModule):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.Hsteps = self.subseq_len -1
        envtask_cfg = self.envtask_cfg

        if envtask_cfg.name != "carla":
            self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
            self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.zeroshot_tasks]

        self.state_processor = StateProcessor(cfg.env_name)
        # self.seen_tasks = envtask_cfg.known_tasks
        # self.unseen_tasks = envtask_cfg.unknown_tasks

        # Losses
        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
        }

        self.outputs = {}
        self.loss_dict = {}
        self.training_step = 0
        self.step = 0
    
    @staticmethod
    def dec_input(states, z, steps, detach = False):
        if detach:
            z = z.clone().detach()
        return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def loss_fn(self, key, index = 1):
        """
        Calcalate loss by loss name 
        """
        return self.loss_fns[key][index]
    
    def grad_clip(self, optimizer):
        if self.step < self.init_grad_clip_step:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.init_grad_clip) 

    @torch.no_grad()
    def get_metrics(self):
        """
        Metrics
        """
        return NotImplementedError
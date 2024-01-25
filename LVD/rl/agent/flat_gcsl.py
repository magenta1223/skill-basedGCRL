
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from easydict import EasyDict as edict
from ...utils import *
from ...models import BaseModel
from ...contrib import *


class Flat_GCSL(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        rl_params = self.policy.get_rl_params()

        self.policy_optim = Adam(
            rl_params['policy'],
        )
        if rl_params['consistency']:
            self.consistency_optims = {  name : {  
                'optimizer' : Adam( args['params'], lr = args['lr'] ),
                'metric' : args['metric']

            }   for name, args in rl_params['consistency'].items() }


            scheduler_params = edict(
                factor = 0.5,
                patience = 10, 
                verbose= True,
            )


            self.consistency_schedulers = {
                k : Scheduler_Helper(v['optimizer'], **scheduler_params, module_name = k) for k, v in self.consistency_optims.items()
            }

            self.schedulers_metric = {
                k : v['metric'] for k, v in self.consistency_optims.items()
            }

            self.consistency_meters = { k : AverageMeter() for k in self.consistency_optims.keys()}

        self.consistency_meter = AverageMeter()

                
        # Alpha
        if isinstance(self.init_alpha, float):
            self.init_alpha = torch.tensor(self.init_alpha)

        pre_init_alpha = simpl_math.inverse_softplus(self.init_alpha)

        if self.auto_alpha:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=self.alpha_lr)
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.n_step = 0
        self.init_grad_clip_step = 0
        self.init_grad_clip = 5.0

        self.sample_time_logger = AverageMeter()

        try:
            self.save_prev_module()
            self.check_delta = True
        except:
            self.check_delta = False

    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def update(self, step_inputs):
        self.train()


        batch = self.buffer.sample(self.rl_batch_size)
        self.episode = step_inputs['episode']
        self.n_step += 1

        self.stat = edict()

        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        self.update_consistency(batch)
        # ------------------- Alpha ------------------- # 


        return self.stat

    def update_consistency(self, batch):

        consistency_losses = self.policy.dist(batch, mode = "consistency")
        consistency_loss = self.aggregate_values(consistency_losses)

        for module_name, optimizer in self.consistency_optims.items():
            optimizer['optimizer'].zero_grad()
            
        consistency_loss.backward()

        for module_name, optimizer in self.consistency_optims.items():
            self.grad_clip(optimizer['optimizer'])
            optimizer['optimizer'].step()

        self.stat.update(consistency_losses)
    
    
    @staticmethod
    def aggregate_values(loss_dict):
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss
    

    def warmup_Q(self, step_inputs):
        # self.train()
        self.stat = {}
        self.episode = step_inputs['episode']

        # # orig : 200 
        for _ in range(int(self.q_warmup)):
            self.update(step_inputs)
            # ------------------- Alpha ------------------- #         
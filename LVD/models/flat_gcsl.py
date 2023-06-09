import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict

class Flat_GCSL(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_amp = True
        self.step = 0
        self.Hsteps = self.subseq_len -1
        self.joint_learn = True

        # prior = SequentialBuilder(cfg.prior)
        # highlevel_policy = SequentialBuilder(cfg.high_policy)
        
        policy = SequentialBuilder(cfg.policy)
        
        self.prior_policy = PRIOR_WRAPPERS['flat_gcsl'](
            # skill_prior = prior,
            policy = policy,
            tanh = self.tanh,
            cfg = cfg,
        )

        # optimizer
        # self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        self.optimizers = {
            "prior_policy" : {
                "optimizer" : RAdam( self.prior_policy.parameters(), lr = self.lr ),
                "metric" : "Rec_skill"
            }, 
        }


        self.outputs = {}
        self.loss_dict = {}

        self.step = 0

    @torch.no_grad()
    def get_metrics(self):
        """
        Metrics
        """
        self.loss_dict['metric'] = self.loss_dict['Rec_skill']

        

    def forward(self, batch):
        states, actions, G = batch.states, batch.actions, batch.G

        # skill prior
        self.outputs =  self.prior_policy(batch)

        # Outputs
        self.outputs['skill'] = actions


    def compute_loss(self, batch):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(self.outputs['policy_skill'], batch.actions)
        loss = recon
        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
        }       


        return loss
    
    def __main_network__(self, batch, validate = False):
        self(batch)
        loss = self.compute_loss(batch)

        if not validate:
            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()

    def optimize(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)
        self.get_metrics()

        return self.loss_dict

    @torch.no_grad()
    def validate(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch, validate= True)
        self.get_metrics()

        return self.loss_dict
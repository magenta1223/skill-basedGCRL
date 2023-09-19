import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
import numpy as np
from ..contrib import update_moving_average
from copy import deepcopy


class AdvQueue:
    def __init__(self, max_len = 50_000) -> None:
        self.max_len = max_len
        self.__queue__ = []
        
    def enqueue(self, advantages):
        self.__queue__.extend(advantages)
        if len(self.__queue__) >= self.max_len :
            self.__queue__ = self.__queue__[ - self.max_len: ]

    def get_threshold(self, threshold):
        return np.percentile(np.array(self.__queue__), threshold)
    
    def mean(self):
        return np.mean(self.__queue__)


class WGCSL(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        

        self.use_amp = True
        self.step = 0
        self.target_update_freq = 20
        self.joint_learn = True

        # prior = SequentialBuilder(cfg.prior)
        # highlevel_policy = SequentialBuilder(cfg.high_policy)
        
        policy = SequentialBuilder(cfg.policy)
        self.q_function = SequentialBuilder(cfg.q_function)
        self.target_q_function = SequentialBuilder(cfg.q_function)

        self.target_q_function.load_state_dict(self.q_function.state_dict())
        
        self.threshold = 0
        self.eps_min = 0.05
        self.baw_delta= 0.15
        self.baw_max = 80

        self.prior_policy = PRIOR_WRAPPERS['flat_gcsl'](
            # skill_prior = prior,
            policy = policy,
            tanh = False,
            cfg = cfg,
        )

        # optimizer
        self.adv_que = AdvQueue(50000)

        self.optimizers = {
            "prior_policy" : {
                "optimizer" : RAdam( self.prior_policy.parameters(), lr = self.lr),
                "metric" : "Rec_skill"
            }, 

            "value" : {
                "optimizer" : RAdam( self.q_function.parameters(), lr = self.lr),
                "metric" : None
            }, 

        }


        self.outputs = {}
        self.loss_dict = {}

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]

        self.step = 0

        self.rl_step = 0

    @torch.no_grad()
    def get_metrics(self, batch):
        """
        Metrics
        """
        self.loss_dict['recon'] = self.loss_fn('recon')(self.outputs['policy_action'], batch.actions)
        self.loss_dict['metric'] = self.loss_dict['Rec_skill']

        

    def forward(self, batch):
        states, actions, G = batch.states, batch.actions, batch.G

        # skill prior
        self.outputs =  self.prior_policy(batch)

        # Outputs
        self.outputs['actions'] = actions


    def compute_loss(self, batch):
        # ----------- SPiRL -------------- # 
        
        # weights = batch.drw * batch.weights
        recon = self.loss_fn('recon')(self.outputs.policy_action, batch.actions, batch.weights)

        recon2 = self.loss_fn('recon')(self.outputs.policy_action, batch.actions, batch.epd_adv_1)


        # recon = self.loss_fn('prior')(
        #     batch.actions,
        #     self.outputs.policy_action_dist,
        #     tanh = False
        # ).mean()
        
        loss = recon
        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "OnlyADv" : recon2.item()
        }       


        return loss


    @torch.no_grad()
    def compute_target_q(self, batch):
        # 아니 시~발 여기서 batch를 그냥 쳐넣으면 어떻게해요
        next_batch = edict({**batch})
        next_batch['states'] = batch.next_states
        
        actions = self.prior_policy(next_batch).policy_action
        q_input = torch.cat((batch.next_states, actions, batch.G), dim = -1)
        target_q = self.target_q_function(q_input).squeeze(-1) 

        return batch.reward + (1 - batch.done) * self.discount * target_q 
    
    @torch.no_grad()
    def calcualate_advantage(self, batch):

        target_q = self.compute_target_q(batch)

        actions = self.prior_policy(batch).policy_action
        q_input = torch.cat((batch.states, actions, batch.G), dim = -1)
        value = self.q_function(q_input).squeeze(-1) 

        adv = target_q - value
        exp_adv = torch.exp(adv)
        # if self.clip_score is not None:
        weights = torch.clamp(exp_adv, max=10)

        return weights, exp_adv, adv

    def __main_network__(self, batch, validate = False):
        if not validate:
            self.step += 1

        # Offline RL 
        
        # Target Value 
        target_q = self.compute_target_q(batch)

        # Value
        q_input = torch.cat((batch.states, batch.actions, batch.G), dim = -1)
        q = self.q_function(q_input).squeeze(-1)

        value_loss = self.loss_fn("recon")(q, target_q)

        # Update Value 
        if not validate:
            self.optimizers['value']['optimizer'].zero_grad()
            value_loss.backward()
            self.optimizers['value']['optimizer'].step()
        
        # advantage
        weights, exp_adv, adv = self.calcualate_advantage(batch)
        self.adv_que.enqueue(adv.detach().cpu().numpy().tolist())

        # eps_adv = exp_adv.clone()

        self.threshold = min(self.threshold + self.baw_delta, self.baw_max)
        threshold = self.adv_que.get_threshold(self.threshold)

        # eps_adv[ exp_adv >= threshold ] = 1
        # eps_adv[ exp_adv < threshold ] = 0.05

        eps_adv = torch.where(adv >= threshold, 1, 0.05).to(exp_adv.device)

        batch['epd_adv_1'] =  torch.where(adv >= threshold, 1, 0).to(exp_adv.device)
        # weights = batch.drw * weights * eps_adv
        batch['weights'] = batch.drw * weights * eps_adv

        # Update policy
        self(batch)
        loss = self.compute_loss(batch)

        if not validate:
            self.optimizers['prior_policy']['optimizer'].zero_grad()
            loss.backward()
            self.optimizers['prior_policy']['optimizer'].step()

            # for _, optimizer in self.optimizers.items():
            #     optimizer['optimizer'].zero_grad()
            # loss.backward()
            # for _, optimizer in self.optimizers.items():
            #     optimizer['optimizer'].step()


        self.loss_dict['value_error'] = value_loss.item()

        # soft update
        if (self.step + 1) % self.target_update_freq == 0:
            update_moving_average(self.target_q_function, self.q_function, beta = 0.05)


    def optimize(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)
        self.get_metrics(batch)

        return self.loss_dict

    @torch.no_grad()
    def validate(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch, validate= True)
        self.get_metrics(batch)

        return self.loss_dict
        
    def set_buffer(self, buffer):
        self.buffer= buffer

    def update(self, step_inputs):
        batch = self.buffer.sample(self.rl_batch_size)
        batch['G'] = step_inputs['G'].repeat(self.rl_batch_size, 1).to(self.device)
        self.episode = step_inputs['episode']
        # self.n_step += 1

        self.step += 1 
        
        # batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)
        self.get_metrics(batch)

        # 
        # self.stat = deepcopy(self.loss_dict)

        return self.loss_dict
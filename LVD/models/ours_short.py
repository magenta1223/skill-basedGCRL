import torch
from torch.optim import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from copy import deepcopy
import math 
from .ours import Ours_Model

class Ours_Shortskill(Ours_Model):
    """
    Ablation for skill length shorter than subgoal horizon
    H = 1,  subgoal horizon = 10 in table 5
    H = 5,  subgoal horizon = 10 in table 5
    H = 10, subgoal horizon = 20 in table 6 (2-skill step subgoal)
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # for easy implementation 
        # cfg.subseq_len = cfg.skill_len + 1
        self.Hsteps = cfg.skill_len
        
        learning_mode_choices = ['only_skill', 'sanity_check', 'skill_reg', 'only_bc']
        assert cfg.learning_mode in learning_mode_choices, f"Invalid learning mode. Valid choices are {learning_mode_choices}"
        
        # state enc/dec  
        state_encoder = Multisource_Encoder(cfg.state_encoder)
        state_decoder = Multisource_Decoder(cfg.state_decoder)

        # prior policy submodules
        inverse_dynamics = InverseDynamicsMLP(cfg.inverse_dynamics)


        skill_step_goal_generator = SequentialBuilder(cfg.skill_step_goal_generator)


        prior = SequentialBuilder(cfg.prior)
        if not cfg.manipulation and cfg.testtest:
            cfg.flat_dynamics.in_feature = cfg.latent_state_dim //2  + cfg.skill_dim
        flat_dynamics = SequentialBuilder(cfg.flat_dynamics)
        dynamics = SequentialBuilder(cfg.dynamics)

        if cfg.manipulation:
            diff_decoder = SequentialBuilder(cfg.diff_decoder)
        else:
            diff_decoder = torch.nn.Identity()
        
        if cfg.learning_mode == "only_skill":
            high_policy = SequentialBuilder(cfg.high_policy)
        else:
            high_policy = None
            
        self.prior_policy = PRIOR_WRAPPERS['ours_short'](
            # components  
            skill_prior = prior,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            skill_step_goal_generator = skill_step_goal_generator,
            dynamics = dynamics,
            flat_dynamics = flat_dynamics,
            diff_decoder = diff_decoder,
            high_policy= high_policy,
            
            # configures
            cfg = cfg,
            ema_update = True,
        )
        


        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        self.goal_encoder = SequentialBuilder(cfg.goal_encoder)

        self.prev_goal_encoder = deepcopy(self.goal_encoder)
        self.random_goal_encoder = SequentialBuilder(cfg.goal_encoder)


        # prev epoch skill encoder
        # hard updated on epoch basis 
        self.prev_skill_encoder = deepcopy(self.skill_encoder)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.prior_policy.skill_prior.parameters(), lr = self.lr ),
                "metric" : "Prior_S",
            }, 
            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill",
                # "metric" : "skill_metric"
                "wo_warmup" : True
            }, 
            "invD" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.inverse_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill",
                # "metric" : "Prior_GC"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.dynamics.parameters()},
                    {"params" : self.prior_policy.flat_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill",
            }, 
            "f" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.skill_step_goal_generator.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill",
            }, 
            
            "state" : {
                "optimizer" : RAdam([
                    {'params' : self.prior_policy.state_encoder.parameters()},
                    {'params' : self.prior_policy.state_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "Rec_state",
                "wo_warmup" : True
            },
            "diff" : {
                "optimizer" : RAdam([
                    {'params' : self.prior_policy.diff_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "diff_loss",
            },
            "goal" : {
                "optimizer" : RAdam([
                    {'params' : self.goal_encoder.parameters()},
                    # {'params' : self.goal_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : None,
                "wo_warmup" : True
            },
        }
        
        if cfg.learning_mode == "only_skill":
            self.optimizers["high_policy"] ={
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.high_policy.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill",
            }
        

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]
        self.c = 0
        self.render = False
        self.do_rollout = False

        self.rnd_mu = None
        self.rnd_std = None
        
        self.g_mu = None
        self.g_std = None
    
    
    def __main_network__(self, batch, validate = False):
        # cut
        # batch = edict( {k : v[] for k, v in batch.items()})
        batch['subgoal'] = batch['states'][:, self.subseq_len -1]
        batch['states'] = batch['states'][:, :self.Hsteps+1]
        batch['actions'] = batch['actions'][:, :self.Hsteps]
        
        self(batch)
        loss = self.compute_loss(batch)
        if not validate:
            self.training_step += 1
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()

            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                # self.grad_clip(optimizer['optimizer'])
                optimizer['optimizer'].step()

            # ------------------ Rollout  ------------------ #
            training = deepcopy(self.training)
            self.eval()
            if self.do_rollout:
                self.rollout(batch)
    
            if training:
                self.train()
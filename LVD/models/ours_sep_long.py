import torch
from torch.optim import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from copy import deepcopy
import math 
from .ours_sep import GoalConditioned_Diversity_Sep_Model

class Ours_LongSkill(GoalConditioned_Diversity_Sep_Model):
    """
    Ablation for longer skill horizon (20)    
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        learning_mode_choices = ['only_skill', 'sanity_check', 'skill_reg', 'only_bc']
        assert cfg.learning_mode in learning_mode_choices, f"Invalid learning mode. Valid choices are {learning_mode_choices}"
        
        # state enc/dec  
        state_encoder = Multisource_Encoder(cfg.state_encoder)
        state_decoder = Multisource_Decoder(cfg.state_decoder)

        # prior policy submodules
        inverse_dynamics = InverseDynamicsMLP(cfg.inverse_dynamics)


        subgoal_generator = SequentialBuilder(cfg.subgoal_generator)


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
            
            
            
        self.prior_policy = PRIOR_WRAPPERS['ours_long'](
            # components  
            skill_prior = prior,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            subgoal_generator = subgoal_generator,
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
                    {"params" : self.prior_policy.subgoal_generator.parameters()},
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
        
    def compute_loss(self, batch):
        # 생성된 data에 대한 discount 
        weights = batch.weights

        # ----------- Skill Recon & Regularization -------------- # 
        recon = self.loss_fn('recon')(self.outputs['actions_hat'], batch.actions, weights)
        reg = (self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']) * weights).mean()

        # ----------- State/Goal Conditioned Prior -------------- # 
        prior = (self.loss_fn('prior')(
            self.outputs['z'],
            self.outputs['prior'], # distributions to optimize
            self.outputs['z_normal'],
            tanh = self.tanh
        ) * weights).mean()

        if self.mode_drop:
            invD_loss = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['post_detach']) * weights).mean()
        else:
            invD_loss = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']) * weights).mean()


        # ----------- Dynamics -------------- # 
        flat_D_loss = self.loss_fn('recon')(
            self.outputs['flat_D'],
            self.outputs['flat_D_target'],
            weights
        ) # 1/skill horizon  
        
        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target'],
            weights
        )         

        # ----------- subgoal generator -------------- #  
        
        
        if self.learning_mode ==  "only_skill":
            # no subgoal generator
            reg_term = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['policy_skill']).mean() 
            F_loss = reg_term
                   
        elif self.learning_mode ==  "sanity_check":
            sanity_check = self.loss_fn('recon')(self.outputs['subgoal_D'], self.outputs['subgoal_D_target'], weights)
            subgoal_bc =  self.loss_fn('recon')(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights)
            # reg_term = (self.loss_fn('reg')(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD
            F_loss = sanity_check + subgoal_bc #+ reg_term
            
        elif self.learning_mode ==  "skill_reg":
            subgoal_bc =  self.loss_fn('recon')(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights)
            reg_term = (self.loss_fn('reg')(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() 
            F_loss = reg_term + subgoal_bc #+ reg_term
            
        else:
            subgoal_bc =  self.loss_fn('recon')(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights)
            F_loss = subgoal_bc
        
        r_int = self.loss_fn('recon')(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights)

        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states'], weights) # ? 

        if self.manipulation:
            diff_loss = self.loss_fn("recon")(
                self.outputs['diff'], 
                self.outputs['diff_target'], 
                weights
            )
        else:
            diff_loss = torch.tensor([0]).cuda()

        
        
        # tanh라서 logprob에 normal 필요할 수도
        # goal_recon = self.loss_fn("recon")(self.outputs['G_hat'], batch.G, weights)
        # goal_reg = self.loss_fn("reg")(self.outputs['goal_dist'], get_fixed_dist(self.outputs['goal_dist'].sample().repeat(1,2), tanh = self.tanh)).mean() * 1e-5

        goal_recon = self.loss_fn("recon_orig")(self.outputs['target_goal_embedding'], self.outputs['goal_embedding'])


        if self.only_flatD:
            D_loss = torch.tensor([0]).cuda()

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon # + skill_logp + goal_logp 
        # loss = recon + reg * self.reg_beta + prior + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon + goal_reg # + skill_logp + goal_logp 


        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "D" : D_loss.item(),
            "flat_D" : flat_D_loss.item(),
            "F_loss" : F_loss.item(),
            "r_int" : r_int.item(),
            "r_int_f" : self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights).item(),
            # "r_int_D" : r_int_D.item() / self.weight.D if self.weight.D else 0,
            # "F_skill_kld" : (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean().item(),
            "KL_F_invD" : (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean().item(),
            "KL_F_z" : (self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']) * weights).mean().item(),
            "diff_loss" : diff_loss.item(),
            "Rec_state" : recon_state.item(),
            "Rec_goal" : goal_recon.item(),
        }       

        return loss
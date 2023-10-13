import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict

# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 

class SkiMo_Model(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        ## skill prior module
        state_encoder = SequentialBuilder(cfg.state_encoder)
        state_decoder = SequentialBuilder(cfg.state_decoder)
        prior = SequentialBuilder(cfg.prior)
        dynamics = SequentialBuilder(cfg.dynamics)
        highlevel_policy = SequentialBuilder(cfg.high_policy)

        reward_function = SequentialBuilder(cfg.reward_function)

        self.prior_policy = PRIOR_WRAPPERS['skimo'](
            skill_prior = prior,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            dynamics = dynamics,
            highlevel_policy = highlevel_policy,
            reward_function= reward_function,
            tanh = self.tanh,
            cfg = cfg
        )

        ## skill encoder
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)

        ## closed-loop skill decoder
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.skill_prior.parameters()},
                    {"params" : self.prior_policy.highlevel_policy.parameters()}
                ], lr = self.lr ),
                "metric" : "Prior_S"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "D"
            }, 
            'state' : {
                "optimizer" : RAdam([
                        {'params' : self.prior_policy.state_encoder.parameters()},
                        {'params' : self.prior_policy.state_decoder.parameters()},
                    ],            
                    lr = self.lr
                    ),
                "metric" : "Rec_state",
                "wo_warmup" : True
            },

            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                # "metric" : "Rec_skill"
                "metric" : "Rec_skill",
                "wo_warmup" : True
            }
        }

    
    @torch.no_grad()
    def get_metrics(self):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        with torch.no_grad():
            # KL (post || state-conditioned prior)
            self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean().item()
            if self.gc:
                self.loss_dict['Policy_loss']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['policy_skill']).mean().item()
            else:
                self.loss_dict['Policy_loss']  = 0
                

            # dummy metric 
            self.loss_dict['metric'] = self.loss_dict['Prior_S']
            

    def forward(self, batch):

        states, actions = batch.states, batch.actions

        # skill Encoder 
        enc_inputs = torch.cat( (actions, states.clone()[:,:-1]), dim = -1)
        q = self.skill_encoder(enc_inputs)[:, -1]
        q_clone = q.clone().detach()
        q_clone.requires_grad = False
        
        post = get_dist(q, tanh = self.tanh)
        post_detach = get_dist(q_clone, tanh = self.tanh)
        fixed_dist = get_fixed_dist(q_clone, tanh = self.tanh)

        if self.tanh:
            skill_normal, skill = post.rsample_with_pre_tanh_value()
            self.outputs['z'] = skill.clone().detach()
            self.outputs['z_normal'] = skill_normal.clone().detach()
        else:
            skill = post.rsample()
            skill_normal = None
            self.outputs['z'] = skill.clone().detach()
            self.outputs['z_normal'] = skill_normal
        
        # Skill Decoder 
        if self.manipulation:
            decode_inputs = self.dec_input(states[:, :, :self.n_pos].clone(), skill, self.Hsteps)
        else:
            decode_inputs = self.dec_input(states.clone(), skill, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        
        batch['skill'] = skill

        # skill prior
        prior_outputs =  self.prior_policy(batch)
        
        self.outputs.update(prior_outputs)


        # Outputs
        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['skill_hat'] = skill_hat
        self.outputs['skill'] = actions


    def compute_loss(self, batch):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(self.outputs['skill_hat'], batch.actions)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        
        prior = self.loss_fn('prior')(
            self.outputs['z'],
            self.outputs['prior'], # distributions to optimize
            self.outputs['z_normal'],
            tanh = self.tanh
        ).mean()

        policy_loss = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['policy_skill']).mean()

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target']
        )  
        
        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 


        # ----------- Add -------------- # 
        loss = recon + reg * self.reg_beta  + prior + D_loss + recon_state + policy_loss


        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Prior" : prior.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
            "D" : D_loss.item(),
            "Rec_state" : recon_state.item()
        }       

        return loss
    
    def __main_network__(self, batch, validate = False):
        self(batch)
        loss = self.compute_loss(batch)

        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()



    def optimize(self, batch, e):
        # inputs & targets       
        batch = edict({  k : v.cuda()  for k, v in batch.items()})


        self.__main_network__(batch)

        self.get_metrics()
        self.prior_policy.soft_update()

        return self.loss_dict
    
    @torch.no_grad()
    def validate(self, batch, e):
        # inputs & targets       
        batch = edict({  k : v.cuda()  for k, v in batch.items()})


        self.__main_network__(batch, validate= True)
        self.get_metrics()

        return self.loss_dict
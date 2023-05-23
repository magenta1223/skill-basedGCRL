import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict

# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 

class Skimo_Model(BaseModel):
    """
    
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_amp = True
        self.Hsteps = self.subseq_len -1



        ## skill prior module


        state_encoder = SequentialBuilder(cfg.state_encoder)
        state_decoder = SequentialBuilder(cfg.state_decoder)
        prior = SequentialBuilder(cfg.prior)
        dynamics = SequentialBuilder(cfg.dynamics)
        highlevel_policy = SequentialBuilder(cfg.high_policy)


        self.prior_policy = PRIOR_WRAPPERS['skimo'](
            prior_policy = prior,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            dynamics = dynamics,
            highlevel_policy = highlevel_policy,
            tanh = self.tanh,
        )

        ## skill encoder
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)

        ## closed-loop skill decoder
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.prior_policy.parameters()},
                    {"params" : self.prior_policy.highlevel_policy.parameters()}
                ], lr = self.lr ),
                "metric" : "Prior_S"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "D"
                # "metric" : "Prior_GC"
            }, 
            'state' : {
                "optimizer" : RAdam([
                        {'params' : self.prior_policy.state_encoder.parameters()},
                        {'params' : self.prior_policy.state_decoder.parameters()},
                    ],            
                    lr = self.lr
                    ),
                "metric" : "Rec_state"
            },

            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                # "metric" : "Rec_skill"
                "metric" : "skill_metric"
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
            z_normal, z = post.rsample_with_pre_tanh_value()
            # self.outputs['z'] = z.clone().detach()
            # self.outputs['z_normal'] = z_normal.clone().detach()
        else:
            z = post.rsample()
            z_normal = None
            # self.outputs['z'] = z.clone().detach()
        
        # Skill Decoder 
        decode_inputs = self.dec_input(states.clone(), z, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        


        # skill prior
        self.outputs =  self.prior_policy(batch)


        # Outputs
        self.outputs['z'] = z.clone().detach()
        if z_normal is not None:
            self.outputs['z_normal'] = z_normal.clone().detach()
        else:
            self.outputs['z_normal'] = None

        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['skill_hat'] = skill_hat
        self.outputs['skill'] = actions


    def compute_loss(self, batch):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(self.outputs['skill_hat'], batch.actions)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        
        if self.tanh:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

            policy_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['policy_skill'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()


        else:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], 
                tanh = self.tanh
            ).mean()

            policy_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['policy_skill'], # distributions to optimize
                tanh = self.tanh
            ).mean()

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target']
        )  
        

        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 
        z_tilde = self.outputs['states_repr']
        z = self.outputs['states_fixed_dist']
        mmd_loss = compute_mmd(z_tilde, z)

        # ----------- Add -------------- # 
        loss = recon + reg * self.reg_beta  + prior + D_loss + recon_state + policy_loss + mmd_loss


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
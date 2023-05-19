import torch
from torch.optim import *
from ..modules.priors import *
# from ..configs.build import *
from ..modules.base import *
from ..modules.subnetworks import *
from ..utils import *
from ..envs.kitchen import KitchenEnv_GC
from ..contrib.simpl.env.kitchen import KitchenTask
from .base import BaseModel

class StateConditioned_Model(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_amp = True
        self.step = 0
        self.Hsteps = self.subseq_len -1
        self.joint_learn = True

        prior = SequentialBuilder(cfg.prior)
        highlevel_policy = SequentialBuilder(cfg.high_policy)
        
        
        self.skill_prior = PRIOR_WRAPPERS['sc'](
            prior_policy = prior,
            highlevel_policy = highlevel_policy,
            tanh = self.tanh
        )

        ## skill encoder
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)

        ## closed-loop skill decoder
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        # optimizer
        # self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.skill_prior.parameters(), lr = self.lr ),
                "metric" : "Prior_S"
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


        self.outputs = {}
        self.loss_dict = {}

        self.step = 0

    @torch.no_grad()
    def get_metrics(self):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        # KL (post || state-conditioned prior)
        self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean().item()
        self.loss_dict['Policy_loss']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['policy_skill']).mean().item()

        # dummy metric 
        self.loss_dict['metric'] = self.loss_dict['Prior_S']
        

    def forward(self, states, actions, G):
        inputs = dict(
            states = states,
            G = G
        )

        # skill prior
        self.outputs =  self.skill_prior(inputs)

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
            self.outputs['z'] = z.clone().detach()
            self.outputs['z_normal'] = z_normal.clone().detach()
        else:
            z = post.rsample()
            self.outputs['z'] = z.clone().detach()
        
        # Skill Decoder 
        decode_inputs = self.dec_input(states.clone(), z, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        
        # Outputs
        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['skill_hat'] = skill_hat
        self.outputs['skill'] = actions


    def compute_loss(self, skill):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(self.outputs['skill_hat'], skill)
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

        # ----------- Add -------------- # 
        loss = recon + reg * self.reg_beta  + prior + policy_loss

        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Prior" : prior.item(),
            "Policy_loss" : policy_loss.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
        }       


        return loss
    
    def __main_network__(self, states, actions, G, validate = False):
        self(states, actions, G)
        loss = self.compute_loss(actions)

        if not validate:
            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()

    def optimize(self, batch, e):
        # inputs & targets       

        states, actions, G = batch.values()
        states, actions, G = states.float().cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G)
        self.get_metrics()

        return self.loss_dict
    
    def validate(self, batch, e):
        # inputs & targets       

        states, actions, G = batch.values()
        states, actions, G = states.float().cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G, validate= True)
        self.get_metrics()

        return self.loss_dict
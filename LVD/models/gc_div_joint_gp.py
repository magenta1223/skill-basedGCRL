import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
import torch.distributions as torch_dist
from easydict import EasyDict as edict

class GoalConditioned_GoalPrompt_Model(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_amp = False
        self.Hsteps = self.subseq_len -1

        # state encoder
        # state enc/dec
        state_encoder = SequentialBuilder(cfg.state_encoder)
        state_decoder = SequentialBuilder(cfg.state_decoder)

        # prior policy submodules
        inverse_dynamics = InverseDynamicsMLP(cfg.inverse_dynamics)
        subgoal_generator = SequentialBuilder(cfg.subgoal_generator)
        prior = SequentialBuilder(cfg.prior)
        flat_dynamics = SequentialBuilder(cfg.flat_dynamics)
        dynamics = SequentialBuilder(cfg.dynamics)

        if self.robotics:
            # ppc_config = {**cfg.prior_config}
            # ppc_config['in_feature'] = self.state_dim
            prior_proprioceptive = SequentialBuilder(cfg.ppc_config)
        else:
            prior_proprioceptive = None
        
        self.prior_policy = PRIOR_WRAPPERS['gc_div_joint'](
            # components  
            skill_prior = prior,
            skill_prior_ppc = prior_proprioceptive,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            subgoal_generator = subgoal_generator,
            dynamics = dynamics,
            flat_dynamics = flat_dynamics,
            # architecture parameters
            ema_update = True,
            tanh = self.tanh,
        )

        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.inverse_dynamics_policy.prior_policy.parameters(), lr = self.lr ),
                "metric" : None
                # "metric" : "Prior_S"
                # "metric" : "Prior_GC"                
            }, 
            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill"
                # "metric" : "skill_metric"
            }, 
            "invD" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.inverse_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "Prior_GC"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.dynamics.parameters()},
                    {"params" : self.inverse_dynamics_policy.flat_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "D"
                # "metric" : "Prior_GC"

            }, 
            "f" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.subgoal_generator.parameters()},
                ], lr = self.lr ),
                "metric" : "F_skill_GT"
                # "metric" : "Prior_GC"
            }, 
            "gp" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.goal_prompt},
                ], lr = self.lr ),
                "metric" : "F_skill_GT"
            },
            "state" : {
                "optimizer" : RAdam([
                    {'params' : self.prior_policy.state_encoder.parameters()},
                    {'params' : self.prior_policy.state_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "recon_state"
            }

        }

        self.c = 0
    @torch.no_grad()
    def get_metrics(self):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        # KL (post || invD by subgoal from f)
        self.loss_dict['F_skill_GT'] = self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']).mean().item()
    
        # KL (post || state-conditioned prior)
        self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean().item()
        
        # KL (post || goal-conditioned policy)
        self.loss_dict['Prior_GC']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']).mean().item()
        
        # KL (invD || prior)
        self.loss_dict['Prior_GC_S']  = self.loss_fn('reg')(self.outputs['invD'], self.outputs['prior']).mean().item()
        
        # dummy metric 
        self.loss_dict['metric'] = self.loss_dict['Prior_GC']
        
        # subgoal by flat dynamics rollout
        reconstructed_subgoal = self.inverse_dynamics_policy.state_decoder(self.outputs['subgoal_rollout'])
        self.loss_dict['Rec_flatD_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['states'][:, -1, :]).item()
        self.loss_dict['Rec_D_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['subgoal_recon_D']).item()

        # state reconstruction 
        self.loss_dict['recon_state'] = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 

    def forward(self, batch):
        
        states, actions = batch.states, batch.actions
            
        N, T, _ = states.shape

        # skill Encoder 
        enc_inputs = torch.cat( (states.clone()[:,:-1], actions), dim = -1)
        q = self.skill_encoder(enc_inputs)[:, -1]
        q_clone = q.clone().detach()
        q_clone.requires_grad = False
        
        post = get_dist(q, tanh = self.tanh)
        post_detach = get_dist(q_clone, tanh = self.tanh)
        fixed_dist = get_fixed_dist(q_clone, tanh = self.tanh)

        if self.tanh:
            skill_normal, skill = post.rsample_with_pre_tanh_value()
            # Outputs
            z = skill.clone().detach()
            z_normal = skill_normal.clone().detach()

        else:
            skill = post.rsample()
            z_normal = None
            z = skill.clone().detach()
        
        # Skill Decoder 
        decode_inputs = self.dec_input(states.clone(), skill, self.Hsteps)
        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)

        # skill prior
        self.outputs =  self.prior_policy(batch)
        
        # skill prior & inverse dynamics's target
        self.outputs['z'] = z
        self.outputs['z_normal'] = z_normal

        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['actions_hat'] = skill_hat
        self.outputs['actions'] = actions

    def compute_loss(self, batch):

        # ----------- Skill Recon & Regularization -------------- # 
        recon = self.loss_fn('recon')(self.outputs['actions_hat'], batch.actions)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        

        # ----------- State/Goal Conditioned Prior -------------- # 
        if self.subgoal_loss == "prior":
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

            invD_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['invD'], 
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean() 
        else:
            prior = self.loss_fn('reg')(
                self.outputs['post_detach'],
                self.outputs['prior'], # distributions to optimize
            ).mean()

            invD_loss = self.loss_fn('reg')(
                self.outputs['post_detach'],
                self.outputs['invD'], 
            ).mean() 

        # # ----------- Dynamics -------------- # 
        flat_D_loss = self.loss_fn('recon')(
            self.outputs['flat_D'],
            self.outputs['flat_D_target']
        ) 

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target']
        )         
        

        # # ----------- subgoal generator -------------- # 
        r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_target'])
        r_int_D = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_target'])
        r_int = r_int_f + r_int_D
        # r_int = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_D'])

        reg_term = self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']).mean()
        # reg_term = self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']).mean()

        F_loss = r_int + reg_term 

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss

        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 
        z_tilde = self.outputs['states_repr']
        z = self.outputs['states_fixed_dist']
        mmd_loss = compute_mmd(z_tilde, z)
        loss = loss + recon_state + mmd_loss

        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "invD" : invD_loss.item(),
            "D" : D_loss.item(),
            "flat_D" : flat_D_loss.item(),
            "F" : F_loss.item(),
            "F_state" : r_int.item(),
            "r_int_f" : r_int_f.item(),
            "r_int_D" : r_int_D.item(),
            "F_skill_kld" : reg_term.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
            "Rec_state" : recon_state.item(),
            "mmd_loss" : mmd_loss.item()
        }       


        return loss
    
    @torch.no_grad()
    def rollout(self, batch):
 
        result = self.prior_policy.rollout(batch)

        # indexing outlier 
        c = result['c']
        states_rollout = result['states_rollout']
        skill_sampled = result['skill_sampled']         

        dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])
        N, T, _ = dec_inputs.shape
        actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)

        states_novel = torch.cat((batch.states[:, :c+1], states_rollout), dim = 1)
        actions_novel = torch.cat((batch.actions[:, :c], actions_rollout), dim = 1)
        
        self.loss_dict['states_novel'] = states_novel[batch.rollout].detach().cpu()
        self.loss_dict['actions_novel'] = actions_novel[batch.rollout].detach().cpu()
        self.c = c

    def __main_network__(self, batch, validate = False):

        self(batch)
        loss = self.compute_loss(batch)

        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                self.grad_clip(optimizer['optimizer'])
                optimizer['optimizer'].step()

        # ------------------ Rollout  ------------------ #
        self.eval()
        self.rollout(batch)
        
        if self.training:
            self.train()

    def optimize(self, batch, e):
        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)


        self.get_metrics()
        self.prior_policy.soft_update()
        # self.step += 1

        return self.loss_dict
    
    @torch.no_grad()
    def validate(self, batch, e):

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch, validate= True)

        self.get_metrics()
        self.step += 1

        return self.loss_dict
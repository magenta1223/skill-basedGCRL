import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from copy import deepcopy
import cv2


class GoalConditioned_Diversity_Joint_Sep_Model(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_amp = False
        self.Hsteps = self.subseq_len -1

        envtask_cfg = self.envtask_cfg

        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
        self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.target_tasks]


        self.state_processor = StateProcessor(cfg.env_name)
        self.seen_tasks = envtask_cfg.known_tasks

        # state enc/dec
        # state_encoder = SequentialBuilder(cfg.state_encoder)
        # state_decoder = SequentialBuilder(cfg.state_decoder)
        
        state_encoder = Multisource_Encoder(cfg.state_encoder)
        state_decoder = Multisource_Decoder(cfg.state_decoder)


        # prior policy submodules
        inverse_dynamics = InverseDynamicsMLP(cfg.inverse_dynamics)
        subgoal_generator = SequentialBuilder(cfg.subgoal_generator)
        prior = SequentialBuilder(cfg.prior)
        flat_dynamics = SequentialBuilder(cfg.dynamics)
        dynamics = SequentialBuilder(cfg.dynamics)

        if cfg.robotics:
            diff_decoder = SequentialBuilder(cfg.diff_decoder)

        else:
            diff_decoder = torch.nn.Identity()

        # if self.robotics:
        #     prior_proprioceptive = SequentialBuilder(cfg.ppc)
        # else:
        #     prior_proprioceptive = None
        
        self.prior_policy = PRIOR_WRAPPERS['gc_div_joint_sep'](
            # components  
            skill_prior = prior,
            # skill_prior_ppc = prior_proprioceptive,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            subgoal_generator = subgoal_generator,
            dynamics = dynamics,
            flat_dynamics = flat_dynamics,
            diff_decoder = diff_decoder,
            # architecture parameters
            cfg = cfg,
            ema_update = True,
            # tanh = self.tanh,
            # grad_pass_invD = self.grad_pass.invD,
            # grad_pass_D = self.grad_pass.D,
            # diff = self.diff,
        )

        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.prior_policy.skill_prior.parameters(), lr = self.lr ),
                # "metric" : "Rec_skill"
                "metric" : "Prior_S"
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
                    {"params" : self.prior_policy.inverse_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill"
                # "metric" : "Prior_GC"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.dynamics.parameters()},
                    {"params" : self.prior_policy.flat_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill"
                # "metric" : "D",
                # "metric" : "Rec_D_subgoal"
                # "metric" : "Prior_GC"

            }, 
            "f" : {
                "optimizer" : RAdam( [
                    {"params" : self.prior_policy.subgoal_generator.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill"
                # "metric" : "F_skill_GT"
                # "metric" : None
                # "metric" : "Prior_GC"
            }, 
            
            "state" : {
                "optimizer" : RAdam([
                    {'params' : self.prior_policy.state_encoder.parameters()},
                    {'params' : self.prior_policy.state_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "recon_state"
                # "metric" : None
            },
            "diff" : {
                "optimizer" : RAdam([
                    {'params' : self.prior_policy.diff_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "diff_loss"
                # "metric" : "ppc_loss"
            }



        }

        def weighted_mse(pred, target, weights):
            agg_dims = list(range(1, len(pred.shape)))
            mse = torch.pow(pred - target, 2).mean(dim = agg_dims)

            weighted_error = mse * weights
            weighted_mse_loss = torch.mean(weighted_error)
            return weighted_mse_loss

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]


        self.c = 0

    @torch.no_grad()
    def get_metrics(self, batch):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        weights = batch.weights
        self.loss_dict['F_skill_GT'] = (self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']) * weights).mean().item()
    
        # KL (post || state-conditioned prior)
        self.loss_dict['Prior_S']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior'])*weights).mean().item()
        
        # KL (post || goal-conditioned policy)
        self.loss_dict['Prior_GC']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']) * weights).mean().item()
        
        # KL (invD || prior)
        self.loss_dict['Prior_GC_S']  = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['prior'])*weights).mean().item()
        
        # dummy metric 
        self.loss_dict['metric'] = self.loss_dict['Prior_GC']
        
        # subgoal by flat dynamics rollout
        reconstructed_subgoal, _, _ = self.prior_policy.state_decoder(self.outputs['subgoal_rollout'])
        self.loss_dict['Rec_flatD_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['states'][:, -1, :], weights).item()
        self.loss_dict['Rec_D_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['subgoal_recon_D'], weights).item()

        # state reconstruction 
        self.loss_dict['recon_state'] = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states'], weights) # ? 
        self.loss_dict['recon_state_subgoal_f'] = self.loss_fn('recon')(self.outputs['subgoal_recon_f'], self.outputs['states'][:,-1], weights) # ? 

        self.loss_dict['recon_state_orig'] = self.loss_fn('recon_orig')(self.outputs['states_hat'], self.outputs['states']) # ? 




    def forward(self, batch):

        states, actions = batch.states, batch.actions
        
        N, T, _ = states.shape
        skill_states = states.clone()

        # skill Encoder 
        enc_inputs = torch.cat( (skill_states.clone()[:,:-1], actions), dim = -1)
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
            # self.outputs['z'] = skill.clone().detach()
            # self.outputs['z_normal'] = None
            z_normal = None
            z = skill.clone().detach()
        
        # 
        if self.decode_only_ppc:
            decode_inputs = self.dec_input(skill_states[:, :, :self.n_obj].clone(), skill, self.Hsteps)
        else:
            decode_inputs = self.dec_input(skill_states.clone(), skill, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        

        # prior_policy_batch = edict(
        #     states = states,
        #     G = G,
        #     skill = z
        # )

        batch['skill'] = z

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

        invD_loss = (self.loss_fn('prior')(
            self.outputs['z'],
            self.outputs['invD'], 
            self.outputs['z_normal'],
            tanh = self.tanh
        ) * weights).mean() 

        # ----------- Dynamics -------------- # 
        flat_D_loss = self.loss_fn('recon')(
            self.outputs['flat_D'],
            self.outputs['flat_D_target'],
            weights
        ) #* 0.1 # 1/skill horizon  

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target'],
            weights
        )         
        

        # # ----------- subgoal generator -------------- # 
        r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_target'], weights)
        r_int_D = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_target'], weights)
        r_int = r_int_f + r_int_D
        reg_term = (self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']) * weights).mean()


        F_loss = r_int + reg_term 

        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states'], weights) # ? 

        if self.robotics:
            diff_loss = self.loss_fn("recon")(
                self.outputs['diff'], 
                self.outputs['diff_target'], 
                weights
            )
        else:
            diff_loss = torch.tensor([0]).cuda()

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss + recon_state + diff_loss
        
        reg_state_loss = torch.tensor([0])

        if self.mmd:
            z_tilde = self.outputs['states_repr']
            z = self.outputs['states_fixed_dist']
            reg_state_loss = compute_mmd(z_tilde, z)
            loss +=  reg_state_loss * self.state_reg_beta




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
            "Reg_state" : reg_state_loss.item(),
            "state_embed_std" : self.outputs['states_repr'].std(dim = -1).mean().item(),
            "diff_loss" : diff_loss.item()
        }       


        return loss
    
    @torch.no_grad()
    def rollout(self, batch, render = False):
        result = self.prior_policy.rollout(batch)

        # indexing outlier 
        c = result['c']
        states_rollout = result['states_rollout']
        skill_sampled = result['skill_sampled']    
        skills = result['skills']  

        N, T = batch.states.shape[:2]
        # dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])
        skills = torch.cat(( skill_sampled.unsqueeze(1).repeat(1,T-c-1, 1), skills ), dim = 1)


        if self.decode_only_ppc:
            dec_inputs = torch.cat((states_rollout[:,:, :self.n_obj], skills), dim = -1)
        else:
            dec_inputs = torch.cat((states_rollout, skills), dim = -1)


        N, T, _ = dec_inputs.shape
        # T까지는 맞음. 그 이후부터 아님. 
        actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)
        

        states_novel = torch.cat((batch.states[:, :c+1], states_rollout), dim = 1)[batch.rollout].detach().cpu()
        actions_novel = torch.cat((batch.actions[:, :c], actions_rollout), dim = 1)[batch.rollout].detach().cpu()
        seq_indices = batch.seq_index[batch.rollout]
        start_indices = batch.start_idx[batch.rollout]

        if self.only_unseen:

            # unseen tasks인 경우에만 추가, 렌더하고 싶음. 
            if self.seen_tasks is not None:
                # unseen_G_indices = [self.state_processor.state_goal_checker(state_seq[-1]) not in self.seen_tasks for state_seq in states_novel]
                unseen_G_indices = [self.state_processor.state_goal_checker(state_seq[-1]) for state_seq in states_novel]
                unseen_G_indices = torch.tensor([ G not in self.seen_tasks and len(G) >= 4  for G in unseen_G_indices], dtype= torch.bool)
                # unseen_G_indices = torch.tensor([ G in self.seen_tasks and len(G) == 4  for G in unseen_G_indices], dtype= torch.bool)

                if unseen_G_indices.sum():
                    self.loss_dict['states_novel'] = states_novel[unseen_G_indices]
                    self.loss_dict['actions_novel'] = actions_novel[unseen_G_indices]
                    self.loss_dict['seq_indices'] = seq_indices[unseen_G_indices]
                    # self.loss_dict['start_indices'] = start_indices[unseen_G_indices]

                    self.c = c
                    self.loss_dict['c'] = start_indices[unseen_G_indices] + c


            else:
                self.loss_dict['states_novel'] = states_novel
                self.loss_dict['actions_novel'] = actions_novel
                self.loss_dict['seq_indices'] = seq_indices
                self.c = c
                # self.loss_dict['c'] = c
                self.loss_dict['c'] = start_indices + c


            if self.env.name == "kitchen" and unseen_G_indices.sum() and not self.render:
                imgs = []
                task = self.state_processor.state_goal_checker(self.loss_dict['states_novel'][0][-1])
                with self.env.set_task(self.tasks[0]):
                    init_qvel = self.env.init_qvel
                    for state in self.loss_dict['states_novel'][0]:
                        self.env.set_state(state, init_qvel)
                        img = self.env.render(mode= "rgb_array")
                        img = img.copy()
                        cv2.putText(img = img,  text = task, color = (255,0,0),  org = (400 // 2, 400 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)
                        imgs.append(img)
                
                self.render = True
                self.loss_dict['render'] = imgs


        else:
            self.loss_dict['states_novel'] = states_novel
            self.loss_dict['actions_novel'] = actions_novel
            self.loss_dict['seq_indices'] = seq_indices
            self.c = c
            self.loss_dict['c'] = start_indices + c

        # self.loss_dict['states_novel'] 
        # 여기에 unseen task중 뭐가 있는지 확인하면 됨. 



    
    def __main_network__(self, batch, validate = False, rollout = False, render = False):

        self(batch)
        loss = self.compute_loss(batch)

        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                # self.grad_clip(optimizer['optimizer'])
                optimizer['optimizer'].step()

        # ------------------ Rollout  ------------------ #
        training = deepcopy(self.training)
        self.eval()
        if rollout:
            self.rollout(batch, render)
    
        if training:
            self.train()

    def optimize(self, batch, e, rollout = False, render = False):
        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch, rollout = rollout, render = render)
        self.get_metrics(batch)
        self.prior_policy.soft_update()
        return self.loss_dict
    
    @torch.no_grad()
    def validate(self, batch, e):
        batch = edict({  k : v.cuda()  for k, v in batch.items()})
        self.__main_network__(batch, validate= True, rollout = False)
        self.get_metrics(batch)
        self.step += 1

        return self.loss_dict
import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from copy import deepcopy
import cv2
from simpl_reproduce.maze.maze_vis import draw_maze
import numpy as np 
from matplotlib import pyplot as plt
from d4rl.pointmaze.maze_model import WALL

class GoalConditioned_Diversity_Sep_Model(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # state enc/dec  
        state_encoder = Multisource_Encoder(cfg.state_encoder)
        state_decoder = Multisource_Decoder(cfg.state_decoder)

        # prior policy submodules
        inverse_dynamics = InverseDynamicsMLP(cfg.inverse_dynamics)

        if self.sg_dist:
            cfg.subgoal_generator.out_dim *= 2 

        subgoal_generator = SequentialBuilder(cfg.subgoal_generator)


        prior = SequentialBuilder(cfg.prior)
        flat_dynamics = SequentialBuilder(cfg.dynamics)
        dynamics = SequentialBuilder(cfg.dynamics)

        if cfg.manipulation:
            diff_decoder = SequentialBuilder(cfg.diff_decoder)
        else:
            diff_decoder = torch.nn.Identity()
     
        self.prior_policy = PRIOR_WRAPPERS['ours_sep'](
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
        )
        
        # 이거 안해도 됨. 
        # self.skill_dist = Learned_Distribution(cfg.skill_dist)
        # self.goal_dist = Learned_Distribution(cfg.goal_dist)

        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(cfg.skill_encoder)
        self.skill_decoder = DecoderNetwork(cfg.skill_decoder)



        self.goal_encoder = SequentialBuilder(cfg.goal_encoder)
        self.goal_decoder = SequentialBuilder(cfg.goal_decoder)
        self.prev_goal_encoder = deepcopy(self.goal_encoder)

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
            # "entropy_skill" : {
            #     "optimizer" : RAdam([
            #         {'params' : self.skill_dist.parameters()},
            #     ], lr = self.lr
            #     ),
            #     "metric" : "skill_ent"
            # },
            # "entropy_goal" : {
            #     "optimizer" : RAdam([
            #         {'params' : self.goal_dist.parameters()},
            #     ], lr = self.lr
            #     ),
            #     "metric" : "goal_ent"
            # },
            "goal" : {
                "optimizer" : RAdam([
                    {'params' : self.goal_encoder.parameters()},
                    {'params' : self.goal_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "Rec_goal",                
            },
        }

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]
        self.c = 0
        self.render = False
        self.do_rollout = False

    def denormalize(self, x):
        """
        Restore for maze and carla
        """
        if self.env_name == "maze":
            x[..., :2] = (x[..., :2] + 0.5) * 40
            x[..., 2:] = x[..., 2:]  * 10
        else:
            pass
        return x

    @torch.no_grad()
    def get_metrics(self, batch):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        weights = batch.weights
        # 이제 싹다 drop한 마당에 GT skill과 같을 이유가 없음. high-policy의 skill과 invD의 skill이 같으면 됨. 
        self.loss_dict['F_skill_GT'] = (self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']) * weights).mean().item()

        # self.loss_dict['F_skill_GT'] = (self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']) * weights).mean().item()

        # KL (post || state-conditioned prior)
        self.loss_dict['Prior_S']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior'])*weights).mean().item()
        
        # KL (post || goal-conditioned policy)
        self.loss_dict['Prior_GC']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']) * weights).mean().item()
        
        # KL (invD || prior)
        self.loss_dict['Prior_GC_S']  = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['prior'])*weights).mean().item()
        
        # dummy metric 
        self.loss_dict['metric'] = self.loss_dict['Prior_GC']
        
        # subgoal by flat dynamics rollout
        states = self.outputs['states']
        reconstructed_subgoal, _, _ = self.prior_policy.state_decoder(self.outputs['subgoal_rollout'])
        states_hat = self.outputs['states_hat']
        subgoal_recon_f = self.outputs['subgoal_recon_f']
        subgoal_recon_D = self.outputs['subgoal_recon_D']
        subgoal_target_recon = self.outputs['subgoal_target_recon']

        if self.normalize:
            states = self.denormalize(states)
            reconstructed_subgoal = self.denormalize(reconstructed_subgoal)
            states_hat = self.denormalize(states_hat)
            subgoal_recon_f = self.denormalize(subgoal_recon_f)
            subgoal_recon_D = self.denormalize(subgoal_recon_D)

        if self.env_name == "maze":
            self.loss_dict['Rec_flatD_pos'] = self.loss_fn('recon')(reconstructed_subgoal[:, :self.n_pos], states[:, -1, :self.n_pos], weights).item()
            self.loss_dict['Rec_flatD_nonpos'] = self.loss_fn('recon')(reconstructed_subgoal[:, self.n_pos:], states[:, -1, self.n_pos:], weights).item()
            self.loss_dict['Rec_D_pos'] = self.loss_fn('recon')(subgoal_recon_D[:, :self.n_pos], states[:, -1, :self.n_pos], weights).item()
            self.loss_dict['Rec_D_nonpos'] = self.loss_fn('recon')(subgoal_recon_D[:, self.n_pos:], states[:, -1, self.n_pos:], weights).item()
            self.loss_dict['Rec_state_pos'] = self.loss_fn('recon')(states_hat[..., :self.n_pos], states[..., :self.n_pos], weights) # ? 
            self.loss_dict['Rec_state_nonpos'] = self.loss_fn('recon')(states_hat[..., self.n_pos:], states[..., self.n_pos:], weights) # ? 

        else:
            self.loss_dict['Rec_flatD_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, states[:, -1, :], weights).item()
            self.loss_dict['Rec_D_subgoal'] = self.loss_fn('recon')(subgoal_recon_D, states[:, -1, :], weights).item()

        self.loss_dict['recon_state_subgoal_f'] = self.loss_fn('recon')(subgoal_recon_f, states[:,-1], weights) # ? 
        # self.loss_dict['recon_state_subgoal_f'] = self.loss_fn('recon')(subgoal_recon_f, subgoal_target_recon, weights) # ? 

        self.loss_dict['recon_state_orig'] = self.loss_fn('recon_orig')(states_hat, states) # ? 

        # mutual information with prev epoch's module

        enc_inputs = torch.cat((batch.states[:, :-1], batch.actions), dim = -1)
        prev_post, _ = self.prev_skill_encoder.dist(enc_inputs, detached = True)

        prev_goal_dist = self.prev_goal_encoder.dist(batch.G)

        


        self.loss_dict['MI_skill'] = self.loss_fn('reg')(self.outputs['post_detach'], prev_post).mean()
        self.loss_dict['MI_goal'] = self.loss_fn('reg')(self.outputs['goal_dist'], prev_goal_dist).mean()

        # self.loss_dict['MI_skill'] = 



    def forward(self, batch):
        # enc_inputs = torch.cat( (skill_states.clone()[:,:-1], actions), dim = -1)
        # q = self.skill_encoder(enc_inputs)[:, -1]
        # q_clone = q.clone().detach()
        # q_clone.requires_grad = False
        
        # post = get_dist(q, tanh = self.tanh)
        # post_detach = get_dist(q_clone, tanh = self.tanh)

        enc_inputs = torch.cat((batch.states[:, :-1], batch.actions), dim = -1)
        post, post_detach = self.skill_encoder.dist(enc_inputs, detached = True)

        if self.tanh:
            skill_normal, skill = post.rsample_with_pre_tanh_value()
            # Outputs
            z = skill.clone().detach()
            z_normal = skill_normal.clone().detach()

        else:
            skill = post.rsample()
            z_normal = None
            z = skill.clone().detach()

        fixed_dist = get_fixed_dist(skill.repeat(1,2), tanh = self.tanh)


        if self.manipulation:
            decode_inputs = self.dec_input(batch.states[:, :, :self.n_pos], skill, self.Hsteps)
        else:
            decode_inputs = self.dec_input(batch.states, skill, self.Hsteps)
            # decode_inputs = self.dec_input(skill_states[:, :, :self.n_pos].clone(), skill, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)

        if not self.manipulation:
            batch['skill'] = z

        batch['skill'] = skill

        # skill prior
        self.outputs =  self.prior_policy(batch)
        
        # skill prior & inverse dynamics's target
        self.outputs['z'] = z
        self.outputs['z_normal'] = z_normal

        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['actions_hat'] = skill_hat
        self.outputs['actions'] = batch.actions


        # self.outputs['skill_dist'] = self.skill_dist.dist(skill)
        # self.outputs['goal_dist'] = self.goal_dist.dist(batch.G)

        goal_dist = self.goal_encoder.dist(batch.G)
        goal_embedding = goal_dist.rsample()
        goal_hat = self.goal_decoder(goal_embedding)
        self.outputs['G_hat'] = goal_hat
        self.outputs['goal_dist'] = goal_dist


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


        # invD_loss = (self.loss_fn('prior')(
        #     self.outputs['z'],
        #     self.outputs['invD'], 
        #     self.outputs['z_normal'],
        #     tanh = self.tanh
        # ) * weights).mean() 

        # invD_loss = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['post_detach']) * weights).mean()
        invD_loss = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']) * weights).mean()


        # ----------- Dynamics -------------- # 
        flat_D_loss = self.loss_fn('recon')(
            self.outputs['flat_D'],
            self.outputs['flat_D_target'],
            weights
        ) #* self.Hsteps #* 0.1 # 1/skill horizon  

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target'],
            weights
        )         

        if self.negativeSamples:
            invD_random_loss = (self.loss_fn('prior')(
                self.outputs['random_z'],
                self.outputs['invD_random'], 
                self.outputs['random_z_normal'],
                tanh = self.tanh
            ) * weights).mean() 

            D_random_loss = self.loss_fn('recon')(
                self.outputs['D_random'],
                self.outputs['D_random_target'],
                weights
            )         
            
            invD_loss = (invD_loss + invD_random_loss) * 0.5
            D_loss = (D_loss + D_random_loss) * 0.5
        
        # ----------- subgoal generator -------------- #         
        if self.sg_dist:
            r_int_f = (self.loss_fn("prior")(
                self.outputs['subgoal_f_target'],
                self.outputs['subgoal_f_dist']
            ) * weights).mean() * self.weight.f 
        else:
            r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights) * self.weight.f 
        

        # r_int_f = 0
        r_int_D = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_D_target'], weights) * self.weight.D
        r_int = r_int_f + r_int_D
        # r_int = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_f'], weights) * self.weight.D

        # 도달한 state만이 중요하므로 반드시 invD만을 target으로 해야 함. 
        # mode dropping
        # (s, g)에 대한 subgoal이 복수개 존재 시, 가장 유력한 하나만 있으면 됨. 다른건 필요 없다. 
        # reg_term = (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD
        # reg_term = (self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']) * weights).mean() * self.weight.invD

        # 지금은 수렴 정도를 파악. 실제 skill과 계산
        reg_term = (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD
        
        
        F_loss = r_int + reg_term 

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
        # skill_logp = - self.outputs['skill_dist'].log_prob(self.outputs['z'], self.outputs['z_normal']).mean()
        # skill_logp = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['skill_dist']) * weights).mean()
        # # 이것도.. KL로 만들어야 하기는 해야 되는데 .. 딱히 상관 없겠지? 
        # goal_logp = - self.outputs['goal_dist'].log_prob(batch.G).mean()
        
        # skill_ent = self.outputs['skill_dist'].entropy().mean()
        # goal_ent = self.outputs['goal_dist'].entropy().mean() # normal이고, 분산이 너무 작아서 음수로 가버림. AE로 인코딩 해야함. 

        goal_recon = self.loss_fn("recon")(self.outputs['G_hat'], batch.G, weights)


        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon # + skill_logp + goal_logp 

        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "D" : D_loss.item(),
            "flat_D" : flat_D_loss.item(),
            "r_int" : r_int.item(),
            "r_int_f" : self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights).item(),
            # "r_int_D" : r_int_D.item() / self.weight.D if self.weight.D else 0,
            "F_skill_kld" : reg_term.item() / self.weight.invD if self.weight.invD else 0,
            "Rec_state" : recon_state.item(),
            "diff_loss" : diff_loss.item(),
            "Rec_goal" : goal_recon.item()
            # 'skill_ent' : skill_ent.item(),
            # 'goal_ent' : goal_ent.item(),
            # 'skill_scale' : get_scales(self.outputs['skill_dist'])
        }       

        return loss
    
    @torch.no_grad()
    def rollout(self, batch):
        result = self.prior_policy.rollout(batch)
        c = result['c']
        states_rollout = result['states_rollout']
        skill_sampled = result['skill_sampled']    
        skills = result['skills']  

        N, T = batch.states.shape[:2]
        skills = torch.cat(( skill_sampled.unsqueeze(1).repeat(1,T-c-1, 1), skills ), dim = 1)

        if self.manipulation:
            dec_inputs = torch.cat((states_rollout[:,:, :self.n_pos], skills), dim = -1)
        else:
            dec_inputs = torch.cat((states_rollout, skills), dim = -1)
            # dec_inputs = torch.cat((states_rollout[:, :, self.n_pos:].clone(), skills), dim = -1)

        N, T, _ = dec_inputs.shape
        actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)
        
        states_novel = torch.cat((batch.states[:, :c+1], states_rollout), dim = 1)[batch.rollout].detach().cpu()
        actions_novel = torch.cat((batch.actions[:, :c], actions_rollout), dim = 1)[batch.rollout].detach().cpu()
        seq_indices = batch.seq_index[batch.rollout]
        start_indices = batch.start_idx[batch.rollout]

        # if self.only_unseen and self.seen_tasks is not None:
        #     # only for check whether unseen goal in the robotics env is generated. 
        #     # not used for performance measure. 
        #     unseen_G_indices = [self.state_processor.state_goal_checker(state_seq[-1]) for state_seq in states_novel]
        #     unseen_G_indices = torch.tensor([ G not in self.seen_tasks and len(G) >= 4  for G in unseen_G_indices], dtype= torch.bool)

        #     if unseen_G_indices.sum():
        #         self.loss_dict['states_novel'] = states_novel[unseen_G_indices]
        #         self.loss_dict['actions_novel'] = actions_novel[unseen_G_indices]
        #         self.loss_dict['seq_indices'] = seq_indices[unseen_G_indices]
        #         self.c = c
        #         self.loss_dict['c'] = start_indices[unseen_G_indices] + c

        # else:
        #     self.loss_dict['states_novel'] = states_novel
        #     self.loss_dict['actions_novel'] = actions_novel
        #     self.loss_dict['seq_indices'] = seq_indices
        #     self.c = c
        #     self.loss_dict['c'] = start_indices + c

        self.loss_dict['states_novel'] = states_novel
        self.loss_dict['actions_novel'] = actions_novel
        self.loss_dict['seq_indices'] = seq_indices
        self.c = c
        self.loss_dict['c'] = start_indices + c

        if self.normalize:
            # 여기서 다르게 해야함. 
            self.loss_dict['states_novel'] = self.denormalize(self.loss_dict['states_novel'])
            # states_novel[:, :2] = (states_novel[:, :2] + 0.5) * 40

        if not self.render:
            if self.env.name == "kitchen":
                imgs = []
                achieved_goal = self.state_processor.state_goal_checker(self.loss_dict['states_novel'][0][-1])
                imgs = render_from_env(env = self.env, task = self.tasks[0], states = self.loss_dict['states_novel'][0], text = achieved_goal)
                self.render = True
                self.loss_dict['render'] = imgs

            elif self.env.name == "maze": 
                orig_goal = batch.finalG[batch.rollout][0].detach().cpu().numpy() 
                generated_stateSeq = self.loss_dict['states_novel'][0].detach().cpu().numpy()
                if self.normalize:
                    orig_goal = (orig_goal + 0.5) * 40
                    generated_stateSeq = (generated_stateSeq + 0.5) * 40
                generated_goal = generated_stateSeq[-1][:2]                

                markers = [
                    edict(
                        data = orig_goal,
                        params = edict(
                            marker = "x",
                            c = "red",
                            s = 200, 
                            zorder = 5, 
                            linewidths = 4
                        )
                    ),
                    edict(
                        data = generated_goal,
                        params = edict(
                            marker = "x",
                            c = "blue",
                            s = 200, 
                            zorder = 5, 
                            linewidths = 4
                        )
                    ),
                    edict(
                        data = generated_stateSeq[0][:2],
                        params = edict(
                            marker = "o",
                            c = "green",
                            s = 200, 
                            zorder = 5, 
                            linewidths = 4
                        )
                    )
                ]
                
                episodes = [ edict(states = generated_stateSeq) ]

                fig = render_from_env(env = self.env, episodes = episodes, markers = markers)

                self.render = True
                self.loss_dict['render'] = fig   

        return self.loss_dict
        # self.loss_dict['states_novel'] 
        # 여기에 unseen task중 뭐가 있는지 확인하면 됨. 

    def __main_network__(self, batch, validate = False):
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
        if self.do_rollout:
            self.rollout(batch)
    
        if training:
            self.train()

    def optimize(self, batch, e):
        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)
        self.get_metrics(batch)
        self.prior_policy.soft_update()
        return self.loss_dict
    
    @torch.no_grad()
    def validate(self, batch, e):
        batch = edict({  k : v.cuda()  for k, v in batch.items()})
        self.__main_network__(batch, validate= True)
        self.get_metrics(batch)
        self.step += 1

        return self.loss_dict
    

    @torch.no_grad()
    def action_smoothing(self, concat_states, concat_actions):
        """
        Noise 때문에 action으로 state가 적절히 복구가 안되는 상황. skill을 통해 한번 스무딩해줍시다 
        그래도 안되는데 (?)
        """        
        self.eval()

        concat_states = torch.tensor(concat_states).unsqueeze(0).cuda()
        concat_actions = torch.tensor(concat_actions).unsqueeze(0).cuda()
        
        N, T, _ = concat_states.shape
        
        # skill 단위로 잘라줍니다
        smooth_actions = []
        
        for i in range((T // self.Hsteps) + 1):
            skill_states = concat_states[:, i * self.Hsteps : (i+1) * self.Hsteps] .clone()
            actions = concat_actions[:, i * self.Hsteps : (i+1) * self.Hsteps]
            skill_len = actions.shape[1]

            if skill_len == 0:
                continue

            enc_inputs = torch.cat( (skill_states.clone()[:, :skill_len], actions), dim = -1)
            q = self.skill_encoder(enc_inputs)[:, -1]        
            post = get_dist(q, tanh = self.tanh)

            if self.tanh:
                skill_normal, skill = post.rsample_with_pre_tanh_value()
                skill = torch.tanh(q.chunk(2, -1)[0]) * 2
            else:
                z_normal, skill = None, post.rsample()
                skill = torch.tanh(q.chunk(2, -1)[0]) * 2

            if self.manipulation:
                decode_inputs = self.dec_input(skill_states[:, :skill_len, :self.n_pos].clone(), skill, skill_len)
            else:
                decode_inputs = self.dec_input(skill_states[:, :skill_len] .clone(), skill, skill_len)

            N, T = decode_inputs.shape[:2]
            skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1).squeeze(0)
            smooth_actions.append(skill_hat)
        
        return torch.cat(smooth_actions, dim = 0).detach().cpu().numpy()
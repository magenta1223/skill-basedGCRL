import torch
from torch.optim import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from copy import deepcopy


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
        flat_dynamics = SequentialBuilder(cfg.flat_dynamics)
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
            "goal" : {
                "optimizer" : RAdam([
                    {'params' : self.goal_encoder.parameters()},
                    {'params' : self.goal_decoder.parameters()},
                ], lr = self.lr
                ),
                "metric" : "Rec_goal",
                "wo_warmup" : True
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
        # ----------- Common Metrics ----------- #
        weights = batch.weights
        # KL (post || state-conditioned prior)
        self.loss_dict['Prior_S']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior'])*weights).mean().item()
        
        # KL (post || goal-conditioned policy)
        self.loss_dict['Prior_GC']  = (self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']) * weights).mean().item()
                
        # dummy metric 
        self.loss_dict['metric'] = self.loss_dict['Prior_GC']
        
        # subgoal by flat dynamics rollout
        if not self.training:
            self.validation_metric(batch)
        

    @torch.no_grad()
    def validation_metric(self, batch):
        weights = batch.weights

        states = self.outputs['states']
        reconstructed_subgoal, _, _ = self.prior_policy.state_decoder(self.outputs['subgoal_rollout'])
        states_hat = self.outputs['states_hat']
        subgoal_recon_f = self.outputs['subgoal_recon_f']
        subgoal_recon_D = self.outputs['subgoal_recon_D']

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

        # mutual information with prev epoch's module
        enc_inputs = torch.cat((batch.states[:, :-1], batch.actions), dim = -1)
        prev_post, _ = self.prev_skill_encoder.dist(enc_inputs, detached = True)

        prev_goal_dist = self.prev_goal_encoder.dist(batch.G)

        # self.loss_dict['MI_skill'] = self.loss_fn('reg')(self.outputs['post_detach'], prev_post).mean()
        self.loss_dict['MI_skill'] = self.loss_fn('reg')(prev_post, self.outputs['post_detach']).mean()
        self.loss_dict['MI_goal'] = self.loss_fn('reg')(self.outputs['goal_dist'], prev_goal_dist).mean()

        goals = batch.G
        known_goals_score = self.loopMI(goals)

        # known goal에서 나올 수 있는 수준의 MI는? -> 
        # known은 log취하면 normal임 -> mu + std * 1.35 -> exp -> threshold로 

        # MI scale이 개큼.. 
        log_score = known_goals_score.log()

        qauntile = torch.tensor([0.05, 0.95], dtype= log_score.dtype, device= log_score.device)
        min_value, max_value = torch.quantile(log_score, qauntile)
        truncated = log_score[(min_value < log_score) & (log_score < max_value)].exp()
        self.loss_dict['MI_goal_mean'] = truncated.mean()
        self.loss_dict['MI_goal_std'] = truncated.std()





    def forward(self, batch):
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

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)

        # if not self.manipulation:
        #     batch['skill'] = z

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

        # invD_loss = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['post_detach']) * weights).mean()

        if self.invD_only_f:
            invD_loss = torch.tensor([0]).cuda()
        else:
            if self.mode_drop:
                invD_loss = (self.loss_fn('reg')(self.outputs['invD'], self.outputs['post_detach']) * weights).mean()
            else:
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

        # ----------- subgoal generator -------------- #         
        # if self.sg_dist:
        #     r_int_f = (self.loss_fn("prior")(
        #         self.outputs['subgoal_f_target'],
        #         self.outputs['subgoal_f_dist']
        #     ) * weights).mean() * self.weight.f 
        # else:
        #     r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights) * self.weight.f 
        
        # r_int = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_D_target'], weights) * self.weight.D

        r_int = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_f_target'], weights) * self.weight.D
        r_int += self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights) * self.weight.D

        # r_int = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_f_target'], weights) * self.weight.D

        # r_int_f = 0
        # r_int = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_D_target'], weights) * self.weight.D
        
        # 지금은 수렴 정도를 파악. 실제 skill과 계산


        if self.mode_drop:
            # reg_term = (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD
            # reg_term = (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD
            reg_term = (self.loss_fn('prior')(
                self.outputs['skill_sub'],
                self.outputs['invD_detach'], # distributions to optimize
                self.outputs['skill_sub_normal'],
                tanh = self.tanh
            ) * weights).mean()
    
        else:
            # reg_term = (self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']) * weights).mean() * self.weight.invD
            reg_term = (self.loss_fn('prior')(
                self.outputs['skill_sub'],
                self.outputs['invD_detach'], # distributions to optimize
                self.outputs['skill_sub_normal'],
                tanh = self.tanh
            ) * weights).mean()
        



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
        goal_recon = self.loss_fn("recon")(self.outputs['G_hat'], batch.G, weights)
        goal_reg = self.loss_fn("reg")(self.outputs['goal_dist'], get_fixed_dist(self.outputs['goal_dist'].sample().repeat(1,2), tanh = self.tanh)).mean() * 1e-5

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon + goal_reg # + skill_logp + goal_logp 
        # loss = recon + reg * self.reg_beta + prior + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon + goal_reg # + skill_logp + goal_logp 


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
            "F_skill_kld" : (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean().item(),
            "diff_loss" : diff_loss.item(),
            "Rec_state" : recon_state.item(),
            "Rec_goal" : goal_recon.item(),
        }       

        # weights = batch.weights
        # goal_recon = self.loss_fn("recon")(self.outputs['G_hat'], batch.G, weights)
        # goal_reg = self.loss_fn("reg")(self.outputs['goal_dist'], get_fixed_dist(self.outputs['goal_dist'].sample().repeat(1,2), tanh = self.tanh)).mean() * 0

        # loss = goal_recon + goal_reg

        # prev_goal_dist = self.prev_goal_encoder.dist(batch.G)

        


        # self.loss_dict = {
        #     'loss' : loss.item(),
        #     'Rec_goal' : loss.item(),
        #     'metric' : loss.item()
        # }

        # self.loss_dict['MI_goal'] = self.loss_fn('reg')(self.outputs['goal_dist'], prev_goal_dist).mean()


        return loss
    
    @torch.no_grad()
    def rollout(self, batch):
        
        rollout_batch = edict({ k : v[batch.rollout] for k, v in batch.items()})


        result = self.prior_policy.rollout(rollout_batch)
        c = result['c']
        states_rollout = result['states_rollout']
        skills = result['skills']  

        if self.manipulation:
            dec_inputs = torch.cat((states_rollout[:,:, :self.n_pos], skills), dim = -1)
        else:
            dec_inputs = torch.cat((states_rollout, skills), dim = -1)

        N, T, _ = dec_inputs.shape
        actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)
        
        states_novel = torch.cat((rollout_batch.states[:, :c+1], states_rollout), dim = 1).detach().cpu()
        actions_novel = torch.cat((rollout_batch.actions[:, :c], actions_rollout), dim = 1).detach().cpu()
        seq_indices = rollout_batch.seq_index.detach().cpu()
        start_indices = rollout_batch.start_idx.detach().cpu()

        # filter 
        goals = rollout_batch.G
        rollout_goals = self.state_processor.goal_transform(states_rollout[:, -1])
        
        known_goals_score = self.loopMI(goals)
        rollout_goals_score = self.loopMI(rollout_goals)

        # known goal에서 나올 수 있는 수준의 MI는? -> 
        log_score = known_goals_score.log()
        qauntile = torch.tensor([0.05, 0.95], dtype= log_score.dtype, device= log_score.device)
        min_value, max_value = torch.quantile(log_score, qauntile)
        truncated = log_score[(min_value < log_score) & (log_score < max_value)]

        mu, std = truncated.mean(), truncated.std()
        threshold = (mu + std * 1.35).exp()
        indices = rollout_goals_score > threshold
        indices = indices.detach().cpu()
        
        # scores = known_goals_score.softmax(dim = 0)



        # score 기반으로 sampling
        # indices = torch_dist.Categorical(rollout_goals_score).sample((100, )).detach().cpu()

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

        self.loss_dict['states_novel'] = states_novel[indices]
        self.loss_dict['actions_novel'] = actions_novel[indices]
        self.loss_dict['seq_indices'] = seq_indices[indices]
        self.c = c
        self.loss_dict['c'] = start_indices[indices] + c

        # analysis
        # rollout_goals = self.state_processor.goal_transform(states_rollout[:, -1])

        


        if sum(indices):
            if self.normalize:
                self.loss_dict['states_novel'] = self.denormalize(self.loss_dict['states_novel'])

            if not self.render:
                if self.env.name == "kitchen":
                    img_len = max(self.loss_dict['c'][0].item() + self.plan_H - 280, 1)  
                    imgs = []
                    achieved_goal = self.state_processor.state_goal_checker(self.loss_dict['states_novel'][0][-1])
                    imgs = render_from_env(env = self.env, task = self.tasks[0], states = self.loss_dict['states_novel'][0], text = achieved_goal)
                    self.render = True
                    self.loss_dict['render'] = imgs[:-img_len]

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
    
    @torch.no_grad()
    def loopMI(self, goals):
        self.goal_encoder(goals)
        goal_dist = self.goal_encoder.dist(goals)
        goal_emb = goal_dist.sample()
        goal_recon = self.goal_decoder(goal_emb)
        goal_reDist = self.goal_encoder.dist(goal_recon)
        scores = self.loss_fn("reg")(goal_dist, goal_reDist)

        return scores
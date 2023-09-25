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

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]
        self.c = 0
        self.render = False
        self.do_rollout = False

        self.rnd_mu = None
        self.rnd_std = None
        
        self.g_mu = None
        self.g_std = None


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

        # self.loss_dict['MI_skill'] = self.loss_fn('reg')(self.outputs['post_detach'], prev_post).mean()
        self.loss_dict['MI_skill'] = self.loss_fn('reg')(prev_post, self.outputs['post_detach']).mean()

        # goals = batch.G
        # known_goals_score = self.loopMI(goals)

        # # known goal에서 나올 수 있는 수준의 MI는? -> 
        # # known은 log취하면 normal임 -> mu + std * 1.35 -> exp -> threshold로 

        # # MI scale이 개큼.. 
        # log_score = known_goals_score.log()

        # qauntile = torch.tensor([0.05, 0.95], dtype= log_score.dtype, device= log_score.device)
        # min_value, max_value = torch.quantile(log_score, qauntile)
        # truncated = log_score[(min_value < log_score) & (log_score < max_value)].exp()
        # self.loss_dict['MI_goal_mean'] = truncated.mean()
        # self.loss_dict['MI_goal_std'] = truncated.std()
    
    @torch.no_grad()
    def normalize_G(self, G):
        # g_mu, g_std = G.mean(dim = 0), G.std(dim = 0)
        normalized = (G - self.g_mu) / (self.g_std + 1e-5)
        return torch.clamp(normalized, -5, 5)



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



        # goal_dist = self.goal_encoder.dist(batch.G)
        # goal_embedding = goal_dist.rsample()
        # goal_hat = self.goal_decoder(goal_embedding)
        # self.outputs['G_hat'] = goal_hat
        # self.outputs['goal_dist'] = goal_dist

        # obs norm 
        # only known goal 
        # if self.manipulation:
        #     normalized_G = self.normalize_G(batch.G[batch.rollout])
        # else:
        #     # 
        #     states = batch.states[batch.rollout][:, 0]
        #     mu, std = states.mean(dim = 0), states.std(dim = 0)
        #     normalized_states = (states - mu) / (std + 1e-5)
        #     normalized_G = self.normalize_G(batch.G[batch.rollout])
        #     normalized_G = torch.cat((normalized_states, normalized_G), dim = -1)


        # goal_embedding = self.goal_encoder(normalized_G)
        # target_goal_embedding = self.random_goal_encoder(normalized_G)

        
        if self.manipulation:
            goal_embedding = self.goal_encoder(batch.G)
            target_goal_embedding = self.random_goal_encoder(batch.G)
        else:
            ge_input = torch.cat((batch.states[:, 0], batch.G), dim = -1)
            goal_embedding = self.goal_encoder(ge_input)
            target_goal_embedding = self.random_goal_encoder(ge_input)




        self.outputs['goal_embedding'] = goal_embedding
        self.outputs['target_goal_embedding'] = target_goal_embedding


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
        sanity_check = self.loss_fn('recon')(self.outputs['subgoal_D'], self.outputs['subgoal_f'], weights)
        subgoal_state_loss =  self.loss_fn('recon')(self.outputs['subgoal_f'], self.outputs['subgoal_f_target'], weights)
        # reg_term = (self.loss_fn('reg')(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean() * self.weight.invD

        F_loss = sanity_check + subgoal_state_loss #+ reg_term
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

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss + recon_state + diff_loss + goal_recon # + skill_logp + goal_logp 
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
            # "F_skill_kld" : (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean().item(),
            "KL_F_invD" : (self.loss_fn("reg")(self.outputs['invD_sub'], self.outputs['invD_detach']) * weights).mean().item(),
            "KL_F_z" : (self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']) * weights).mean().item(),
            "diff_loss" : diff_loss.item(),
            "Rec_state" : recon_state.item(),
            "Rec_goal" : goal_recon.item(),
        }       

        # if self.advantage:
        #     self.loss_dict['adv'] = adv.mean()



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
        
        states_novel = torch.cat((rollout_batch.states[:, :c+1], states_rollout), dim = 1)
        actions_novel = torch.cat((rollout_batch.actions[:, :c], actions_rollout), dim = 1)
        seq_indices = rollout_batch.seq_index
        start_indices = rollout_batch.start_idx



        # filter 
   


        # print(rollout_goals_score, threshold, indices)



        # known_goals_score = self.loopMI(goals)
        # rollout_goals_score = self.loopMI(rollout_goals)

        # # known goal에서 나올 수 있는 수준의 MI는? -> 
        # log_score = known_goals_score.log()
        # qauntile = torch.tensor([0.05, 0.95], dtype= log_score.dtype, device= log_score.device)
        # min_value, max_value = torch.quantile(log_score, qauntile)
        # truncated = log_score[(min_value < log_score) & (log_score < max_value)]

        # mu, std = truncated.mean(), truncated.std()
        # threshold = (mu + std * 1.35).exp()
        # indices = rollout_goals_score > threshold
        # indices = indices.detach().cpu()
        
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
        
        if self.filter_default:
            indices = self.filter_rollout(result, rollout_batch)
        else:
            indices = self.filter_rollout3(result, rollout_batch)

        self.loss_dict['states_novel'] = states_novel[indices].detach().cpu()
        self.loss_dict['actions_novel'] = actions_novel[indices].detach().cpu()
        self.loss_dict['seq_indices'] = seq_indices[indices].detach().cpu()
        self.c = c
        self.loss_dict['c'] = (start_indices[indices] + c).detach().cpu()

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


        if e < 1:
            # goal mu, std 계산 
            g_mu, g_std = batch.G.mean(dim = 0), batch.G.std(dim = 0)

            if self.g_mu is None:
                self.g_mu = g_mu 
                self.g_std = g_std
            else:
                ratio = 1e-3
                self.g_mu = self.g_mu * (1 - ratio) + g_mu * ratio
                self.g_std = self.g_std * (1 - ratio) + g_std * ratio             





        self.__main_network__(batch)
        self.get_metrics(batch)
        self.prior_policy.soft_update()

        # rnd threshold 
        goal_recon_errors = torch.pow(self.outputs['target_goal_embedding'] - self.outputs['goal_embedding'], 2).mean(dim = -1).log()
        
        mu, std = goal_recon_errors.mean().item(), goal_recon_errors.std().item()


        if self.rnd_mu is None:
            self.rnd_mu = mu 
            self.rnd_std = std
        else:
            ratio = 1e-3
            self.rnd_mu = self.rnd_mu * (1 - ratio) + mu * ratio
            self.rnd_std = self.rnd_std * (1 - ratio) + std * ratio 




        
        # print(f"{self.rnd_mu:.5f}, {self.rnd_std:.5f}")

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
    def filter_rollout(self, result, rollout_batch):
        goals = rollout_batch.G
        N = goals.shape[0]
      
        # 2. get branching point 
        branching_point=  result.c
        
        # 3. get final state of rollout 
        start_indices = rollout_batch.start_idx
        # max_len - start_indices 
        # start + branching + plan_H > max_len일 때 
        # trajectory의 최대길이에 해당하는 state의 index 
        max_indices_rollout = torch.minimum(- start_indices + self.max_seq_len - branching_point, torch.full_like(start_indices, self.plan_H - 1))
        max_indices_rollout = max_indices_rollout.to(dtype = torch.int)
        rollout_goals = self.state_processor.goal_transform(result.states_rollout[list(range(N)), max_indices_rollout])

        
        # goals = self.normalize_G(goals)
        # rollout_goals = self.normalize_G(rollout_goals)
  
        
        if self.manipulation:
            goal_emb = self.goal_encoder(goals)
            goal_emb_random = self.random_goal_encoder(goals)

        else:
            ge_input = torch.cat((rollout_batch.states[:, 0], rollout_batch.goals), dim = -1)
            rnd_ge_input = torch.cat((rollout_batch.states[:, 0], rollout_goals), dim = -1)

            goal_emb = self.goal_encoder(ge_input)
            goal_emb_random = self.random_goal_encoder(rnd_ge_input)


        # Random Network distillation 
        goal_emb = self.goal_encoder(goals)
        goal_emb_random = self.random_goal_encoder(goals)
        known_goals_score = ((goal_emb - goal_emb_random) ** 2).mean(dim = -1)
        
        goal_emb = self.goal_encoder(rollout_goals)
        goal_emb_random = self.random_goal_encoder(rollout_goals)
        rollout_goals_score = ((goal_emb - goal_emb_random) ** 2).mean(dim = -1)

        # known goal에서 나올 수 있는 수준의 MI는? -> 
        log_score = known_goals_score.log()
        qauntile = torch.tensor([0.05, 0.95], dtype= log_score.dtype, device= log_score.device)
        min_value, max_value = torch.quantile(log_score, qauntile)
        truncated = log_score[(min_value < log_score) & (log_score < max_value)]

        mu, std = truncated.mean(), truncated.std()
        threshold = (mu + std * 1.35).exp()
        indices = rollout_goals_score > threshold
        indices = indices.detach().cpu()
        

        return indices
    
    @torch.no_grad()
    def filter_rollout2(self, result, rollout_batch):
        
        # copy 
        result = edict(**result)
        rollout_batch = edict(**rollout_batch)

        # 1. tau_im의 latent state
        states_rollout = result.states_rollout 
        latent_state_rollout = result.latent_states_rollout

        # 2. get branching point 
        branching_point=  result.c
        
        # 3. get final state of rollout 
        start_indices = rollout_batch.start_idx
        # max_len - start_indices 
        # start + branching + plan_H > max_len일 때 
        # trajectory의 최대길이에 해당하는 state의 index 
        max_indices_rollout = torch.minimum(- start_indices + self.max_seq_len - branching_point, torch.full_like(start_indices, self.plan_H - 1))
        max_indices_rollout = max_indices_rollout.to(dtype = torch.int)

        sample_indices = list(range(len(states_rollout)))

        G = self.state_processor.goal_transform(states_rollout[sample_indices, max_indices_rollout])
        # policy를 이용해 검증 
        

        # validation_batch = edict(
        #     ht = latent_state_rollout[:, 0], # branching point부터 시작 ? 아니면 subtraj 처음부터 시작? 
        #     G =  G
        # )

        # validation_states = []
        
        # # rollout horizon  // H 만큼 반복
        # for _ in range(self.plan_H // self.Hsteps):
        #     ht = self.prior_policy.dist(validation_batch, mode = "rollout").ht
        #     validation_batch['ht'] = ht
        #     validation_states.append(ht)
        
        # validation_states = torch.stack(validation_states, dim = 1)
        # validation_indices = max_indices_rollout // self.Hsteps

        # # 시점을 skill 종료 지점으로 맞춰주기 
        # sample_indices_skill_level = list(range(len(validation_states)))
        # latent_G = latent_state_rollout[sample_indices, validation_indices * 10]

        # diffs = torch.pow(validation_states[sample_indices_skill_level, validation_indices] - latent_G, 2).mean(dim = -1)

        validation_batch = edict(
            ht = latent_state_rollout[:, 0], # branching point부터 시작 ? 아니면 subtraj 처음부터 시작? 
            G =  G
        )

        validation_states = []
        
        error = 0
        # rollout horizon  // H 만큼 반복
        for i in range(self.plan_H // self.Hsteps):
            validation_batch['ht'] = latent_state_rollout[:, i]
            ht_executed = self.prior_policy.dist(validation_batch, mode = "rollout").ht

            error += torch.pow(validation_batch.ht - ht_executed, 2).mean(dim = -1)
        
        # validation_states = torch.stack(validation_states, dim = 1)
        # validation_indices = max_indices_rollout // self.Hsteps

        # 시점을 skill 종료 지점으로 맞춰주기 
        # sample_indices_skill_level = list(range(len(validation_states)))
        # latent_G = latent_state_rollout[sample_indices, validation_indices * 10]

        # diffs = torch.pow(validation_states[sample_indices_skill_level, validation_indices] - latent_G, 2).mean(dim = -1)
        diffs = error / max_indices_rollout * 0.1


                
        qauntile = torch.tensor([0.05, 0.95], dtype= diffs.dtype, device= diffs.device)
        min_value, max_value = torch.quantile(diffs, qauntile)
        diffs_trunc = diffs[(min_value < diffs) & (diffs < max_value)]
        
        # outlier threshold 
        threshold = (diffs_trunc.mean() + diffs_trunc.std() * 1.1)
        indices = diffs > threshold

        # 특정 task

        # print(indices.sum().item() / len(indices))
        print(diffs_trunc.mean().item())

        return indices

    @torch.no_grad()
    def filter_rollout3(self, result, rollout_batch):
        goals = rollout_batch.G
        N = goals.shape[0]
      
        # 2. get branching point 
        branching_point=  result.c
        
        # 3. get final state of rollout 
        start_indices = rollout_batch.start_idx
        # max_len - start_indices 
        # start + branching + plan_H > max_len일 때 
        # trajectory의 최대길이에 해당하는 state의 index 
        max_indices_rollout = torch.minimum(- start_indices + self.max_seq_len - branching_point, torch.full_like(start_indices, self.plan_H - 1))
        max_indices_rollout = max_indices_rollout.to(dtype = torch.int)
        rollout_goals = self.state_processor.goal_transform(result.states_rollout[list(range(N)), max_indices_rollout])


        # if self.manipulation:
        #     goals = self.normalize_G(goals)
        #     rollout_goals = self.normalize_G(rollout_goals)

        # else:
        #     # 
        #     states = rollout_batch.states[:, 0]
        #     mu, std = states.mean(dim = 0), states.std(dim = 0)
        #     normalized_states = (states - mu) / (std + 1e-5)

        #     goals = self.normalize_G(goals)
        #     rollout_goals = self.normalize_G(rollout_goals)

        #     goals = torch.cat((normalized_states, goals), dim = -1)
        #     rollout_goals = torch.cat((normalized_states, rollout_goals), dim = -1)





        # Random Network distillation         
        # goal_emb = self.goal_encoder(rollout_goals)
        # goal_emb_random = self.random_goal_encoder(rollout_goals)

        if self.manipulation:
            goal_emb = self.goal_encoder(rollout_goals)
            goal_emb_random = self.random_goal_encoder(rollout_goals)

        else:
            ge_input = torch.cat((rollout_batch.states[:, 0], rollout_goals), dim = -1)

            goal_emb = self.goal_encoder(ge_input)
            goal_emb_random = self.random_goal_encoder(ge_input)


        rollout_goals_score = ((goal_emb - goal_emb_random) ** 2).mean(dim = -1).log()

        # known goal에서 나올 수 있는 수준의 MI는? -> 
        
        # 기준치 테스트 필요 
        threshold = (self.rnd_mu + self.rnd_std * self.std_factor)
        indices = rollout_goals_score > threshold
        indices = indices.detach().cpu()

        
        print(rollout_goals_score.mean().item(), threshold)
            

        return indices
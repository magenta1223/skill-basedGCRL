import torch
from torch.optim import *
# from ..configs.build import *
from ..modules import *
from ..utils import *
from .base import BaseModel
from easydict import EasyDict as edict
from ..contrib import update_moving_average

class RIS(BaseModel):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_amp = True
        self.step = 0
        self.target_update_freq = 20
        self.joint_learn = True

        # prior = SequentialBuilder(cfg.prior)
        # highlevel_policy = SequentialBuilder(cfg.high_policy)
        
        policy = SequentialBuilder(cfg.policy)
        self.q_function = SequentialBuilder(cfg.q_function)
        self.target_q_function = SequentialBuilder(cfg.q_function)

        self.target_q_function.load_state_dict(self.q_function.state_dict())
        

        self.prior_policy = PRIOR_WRAPPERS['ris'](
            # skill_prior = prior,
            policy = policy,
            tanh = False,
            cfg = cfg,
        )

        # optimizer

        self.optimizers = {
            "prior_policy" : {
                "optimizer" : RAdam( self.prior_policy.parameters(), lr = self.lr ),
                "metric" : "Rec_skill"
            }, 

            "value" : {
                "optimizer" : RAdam( self.q_function.parameters(), lr = self.lr ),
                "metric" : "value_error"
            }, 

        }


        self.outputs = {}
        self.loss_dict = {}

        self.loss_fns['recon'] = ['mse', weighted_mse]
        self.loss_fns['recon_orig'] = ['mse', torch.nn.MSELoss()]

        self.step = 0

    @torch.no_grad()
    def get_metrics(self, batch):
        """
        Metrics
        """
        self.loss_dict['recon'] = self.loss_fn('recon')(self.outputs['policy_action'], batch.actions)
        self.loss_dict['metric'] = self.loss_dict['Rec_skill']

        

    def forward(self, batch):
        states, actions, G = batch.states, batch.actions, batch.G

        # skill prior
        self.outputs =  self.prior_policy(batch)

        # Outputs
        self.outputs['actions'] = actions


    def compute_loss(self, batch):
        # ----------- SPiRL -------------- # 
        
        weights = batch.drw * batch.weights
        recon = self.loss_fn('recon')(self.outputs.policy_action, batch.actions, weights)
        
        # recon = self.loss_fn('prior')(
        #     batch.actions,
        #     self.outputs.policy_action_dist,
        #     tanh = False
        # ).mean()
        

        loss = recon
        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
        }       


        return loss


    @torch.no_grad()
    def compute_target_q(self, batch):
        actions = self.prior_policy(batch).policy_action
        q_input = torch.cat((batch.next_states, actions, batch.G), dim = -1)
        target_q = self.target_q_function(q_input).squeeze(-1) 

        return batch.reward + (1 - batch.dones) * self.discount * target_q 
    
    def forward_value(self, batch, mode = "default", subgoal = None ):
        assert mode in ['default', 'to_subgoal', 'to_goal'], "Invalid mode"

        if mode == "default":
            q_input = torch.cat((batch.states, batch.actions, batch.G))
        elif mode == "to_subgoal":
            if subgoal is not None:
                q_input = torch.cat((batch.states, batch.actions, subgoal))
            else:
                q_input = torch.cat((batch.states, batch.actions, batch.subgoals))
        else: # to_goal
            # assert subgoal is not None, "to goal but subgoal is None"
            if subgoal is not None:                
                q_input = torch.cat((subgoal, batch.actions, batch.G))

            else:
                q_input = torch.cat((batch.subgoal, batch.actions, batch.G))


        value = self.q_function(q_input).squeeze(-1)
        return value
    
    def get_cost(self, batch, subgoal):
        state_subgoal_value = self.forward_value(batch, "to_subgoal", subgoal)
        subgoal_goal_value = self.forward_value(batch, "to_goal", subgoal)
        return torch.where(state_subgoal_value > subgoal_goal_value, state_subgoal_value, subgoal_goal_value)

    @torch.no_grad()
    def calcualate_advantage(self, batch, predicted_subgoal):
        # cost function 

        cost = self.get_cost(batch, batch.subgoal)
        cost_pred = self.get_cost(batch, predicted_subgoal)

        adv = torch.exp((cost_pred - cost) * self.scale_lambda)

        return torch.softmax(adv, dim = 0)

    def __main_network__(self, batch, validate = False):
        if not validate:
            self.step += 1

        # ------------------ update value function ------------------ #

        # Target Value 
        target_q = self.compute_target_q(batch)

        # Value
        q_input = torch.cat((batch.states, batch.actions, batch.G), dim = -1)
        q = self.q_function(q_input).squeeze(-1)

        value_loss = self.loss_fn("recon")(q, target_q)

        # Update Value 
        if not validate:
            self.optimizers['value']['optimizer'].zero_grad()
            value_loss.backward()
            self.optimizers['value']['optimizer'].step()        
        # ----------------------------------------------------------- #

        # ----------------- Update High Level Policy ---------------- #
        result = self.prior_policy(batch, "subgoal")

        # subgoal
        subgoal_dist = result.subgoal_dist

        # advantage 계산
        adv = self.calcualate_advantage(batch, subgoal_dist.sample())
        subgoal_loss = subgoal_dist.log_prob(batch.subgoal).mean() * adv

        if not validate:
            self.optimizers['subgoal']['optimizer'].zero_grad()
            subgoal_loss.backward()
            self.optimizers['subgoal']['optimizer'].step()
        # ----------------------------------------------------------- #

        # ---------------------- Update Policy ---------------------- #
        self.outputs =  self.prior_policy(batch)

        # Outputs
        self.outputs['actions'] = batch.actions

        q_input = torch.cat((batch.states, self.outputs.policy_actions, batch.G))
        value = self.q_function(q_input).squeeze(-1)

        prior_regularize = self.loss_fn("reg")( self.outputs.policy_action_dist, self.outputs.subgoal_action_dist ).mean()

        policy_loss = (-value + self.alpha * prior_regularize)


        # loss = self.compute_loss(batch)

        if not validate:
            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            policy_loss.backward()

            for _, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()


        self.loss_dict['value_error'] = value_loss.item()
        # ----------------------------------------------------------- #

        # soft update
        if (self.step + 1) % self.target_update_freq == 0:
            update_moving_average(self.target_q_function, self.q_function, beta = 0.05)
            self.prior_policy.soft_update()


    def optimize(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch)
        self.get_metrics(batch)

        return self.loss_dict

    @torch.no_grad()
    def validate(self, batch, e):
        # inputs & targets       

        batch = edict({  k : v.cuda()  for k, v in batch.items()})

        self.__main_network__(batch, validate= True)
        self.get_metrics(batch)

        return self.loss_dict
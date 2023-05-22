# envs & utils
import gym
import d4rl

import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb
import cv2
# simpl contribs
# from proposed.contrib.simpl.collector import Buffer

# models
import torch.nn as nn
from torch.nn import functional as F

from simpl_reproduce.maze.maze_vis import draw_maze

from LVD.modules import *
from LVD.rl.sac import SAC
from LVD.contrib.simpl.torch_utils import itemize
from LVD.utils import *

from LVD.collector.gcid import LowFixedHierarchicalTimeLimitCollector
from LVD.collector.storage import Buffer_G


seed_everything()

import hydra


### RL Trainer ### 

class RL_Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.set_env()
    
        os.makedirs(self.cfg.weights_path, exist_ok= True)


        # ------------- Logger ------------- #
        wandb.init(
            project = self.cfg.project_name,
            name = self.cfg.wandb_run_name,

            # name = f"LEVEL {str(args.level)}", 
            config = {
                # "alpha" : args.init_alpha,
                # 'policy_lr' : args.policy_lr,
                # # 'prior_policy_lr' : sac_config['prior_policy_lr'],
                # 'target_kl_end' : args.target_kl_end,
                # 'warmup' : 0, #sac_config['warmup'],
                # "pretrained_model" : f"{args.path}",
                # 'q_warmup_steps' : args.q_warmup, 
                # 'precollect' : args.precollect, 
            },
        )
    
    def set_env(self):
        envtask_cfg = self.cfg.envtask_cfg

        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
        
        self.tasks = []
        for task in envtask_cfg.target_tasks:
            task_obj = envtask_cfg.task_cls(task)
            self.tasks.append((str(task_obj), task_obj))


    # fitting 할 때 마다 호출 
    def prep(self):
        # load skill learners and prior policy
        model = self.cfg.skill_trainer.load(self.cfg.skill_weights_path).model
    
        # learnable
        high_policy = model.prior_policy

        # non-learnable
        low_actor = deepcopy(model.skill_decoder.eval())
        skill_prior = deepcopy(model.prior_policy.skill_prior)
        low_actor.requires_grad_(False)
        skill_prior.requires_grad_(False) 

        # build qfs
        qfs = [ SequentialBuilder(self.cfg.q_function)  for _ in range(2)]
        
        # build buffers and collector

        buffer = Buffer_G(self.cfg.state_dim, self.cfg.skill_dim, self.cfg.n_goal, self.cfg.buffer_size, self.cfg.tanh)
        collector = LowFixedHierarchicalTimeLimitCollector(
            self.env,
            # env_name,
            low_actor,
            horizon = self.cfg.subseq_len,
            time_limit= self.cfg.time_limit,
        )
        
        # See rl_cfgs in LVD/configs/common.yaml 
        sac_config = { attr_nm : getattr(self.cfg, attr_nm) for attr_nm in self.cfg.rl_cfgs}
        sac_modules = {
            'policy' : high_policy,
            'prior_policy' : skill_prior, 
            'buffer' : buffer, 
            'qfs' : qfs,            
        }        

        sac_config.update(sac_modules)
        sac = SAC(sac_config).cuda()

        return collector, sac

    def fit(self):
        
        for task_name, task_obj in self.tasks:
            collector, sac = self.prep()

            torch.save({
                "model" : sac,
                # "collector" : collector if env_name != "carla" else None,
                # "task" : task_obj if env_name != "carla" else np.array(task_obj),
                # "env" : env if env_name != "carla" else None,
            }, f"{self.cfg.weights_path}/{task_name}.bin")   


            with self.env.set_task(task_obj):
                state = self.env.reset()
                # 이건 env에 넣을까? 
                # task = state_processor.get_goals(state)
                # print("TASK : ",  state_processor.state_goal_checker(state, env, mode = "goal") )
                # print("TASK : ",  GOAL_CHECKERS[args.env_name](   GOAL_TRANSFORM[args.env_name](state)  ))

                ewm_rwds = 0
                early_stop = 0
                for n_ep in range(self.cfg.n_episode+1):                    
                    log = self.train_policy(collector, sac, n_ep)
                    log['n_ep'] = n_ep
                    log[f'{task_name}_return'] = log['tr_return']
                    del log['tr_return']

                    if (n_ep + 1) % self.cfg.render_period == 0:
                        self.visualize(collector)

                    log = {f"{task_name}/{k}": log[k] for k in log.keys()}
                    wandb.log(log)

                    # clear plot
                    plt.cla()

                    ewm_rwds = 0.8 * ewm_rwds + 0.2 * log[f'tr_return']

                    if ewm_rwds > self.cfg.early_stop_threshold:
                        early_stop += 1
                    else:
                        early_stop = 0
                    
                    if early_stop == 10:
                        print("Converged enough. Early Stop!")
                        break

    def train_policy(self, collector, sac, n_ep):

        log = {}

        # ------------- Collect Data ------------- #
        with sac.policy.expl(), collector.low_actor.expl() : #, collector.env.step_render():
            episode, G = collector.collect_episode(sac.policy)

        if np.array(episode.rewards).sum() == self.cfg.max_reward: # success 
            print("success")

        sac.buffer.enqueue(episode.as_high_episode()) 
        log['tr_return'] = sum(episode.rewards)

        
        if sac.buffer.size < self.cfg.rl_batch_size or n_ep < self.cfg.precollect:
            return log
        
        
        if n_ep == self.cfg.precollect:
            step_inputs = dict(
                G = G,
                episode = n_ep,
            )
            # Q-warmup
            print("Warmup Value function")
            sac.warmup_Q(step_inputs)

        for _ in range(max(self.n_step(episode), 1)):
            step_inputs = dict(
                G = G,
                episode = n_ep,
            )
            stat = sac.step(step_inputs)

        log.update(itemize(stat))

        return log


    def visualize(self, collector):
        pass
        # if env_name == "maze":
        #     log[f'policy_vis'] = draw_maze(plt.gca(), env, list(self.buffer.episodes)[-20:])
        # elif env_name == "kitchen":
        #     imgs = render_task(env, env_name, self.policy, low_actor, tanh = model.tanh)
        #     imgs = np.array(imgs).transpose(0, 3, 1, 2)
        #     if args.env_name == "maze":
        #         fps = 100
        #     else:
        #         fps = 50
        #     log[f'rollout'] = wandb.Video(np.array(imgs), fps=fps)

        # # check success rate by 20 rollout 
        # with self.policy.expl(), collector.low_actor.expl() : #, collector.env.step_render():
        #     episode, G = collector.collect_episode(self.policy)
        
        # if np.array(episode.rewards).sum() == args.max_reward: # success 
        #     print("success")
        

    def n_step(self, episode):
        return int(self.cfg.reuse_rate * len(episode) / self.cfg.batch_size)
# envs & utils
import gym
import d4rl
import os
from copy import deepcopy
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from .sac import SAC
from ..modules import *
from ..contrib.simpl.torch_utils import itemize
from ..utils import *
from ..collector import GC_Hierarchical_Collector, GC_Buffer

from simpl_reproduce.maze.maze_vis import draw_maze

seed_everything()

class RL_Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.set_env()
        os.makedirs(self.cfg.weights_path, exist_ok= True)
        self.rl_cfgs = { attr_nm : getattr(self.cfg, attr_nm) for attr_nm in self.cfg.rl_cfgs}
        # ------------- Logger ------------- #
        wandb.init(
            project = self.cfg.project_name,
            name = self.cfg.wandb_run_name,

            # name = f"LEVEL {str(args.level)}", 
            config = self.rl_cfgs,
        )
    
    def set_env(self):
        envtask_cfg = self.cfg.envtask_cfg

        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)

        self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.target_tasks]

    def prep(self):
        # load skill learners and prior policy
        model = self.cfg.skill_trainer.load(self.cfg.skill_weights_path, self.cfg).model
    
        # learnable
        high_policy = deepcopy(model.prior_policy)

        # non-learnable
        low_actor = deepcopy(model.skill_decoder.eval())
        skill_prior = deepcopy(model.prior_policy.skill_prior)
        low_actor.requires_grad_(False)
        skill_prior.requires_grad_(False) 

        qfs = [ SequentialBuilder(self.cfg.q_function)  for _ in range(2)]
        buffer = GC_Buffer(self.cfg.state_dim, self.cfg.skill_dim, self.cfg.n_goal, self.cfg.buffer_size, self.env.name, model.tanh).to(high_policy.device)
        collector = GC_Hierarchical_Collector(
            self.env,
            low_actor,
            horizon = self.cfg.subseq_len -1,
            time_limit= self.cfg.time_limit,
        )

        sac_modules = {
            'policy' : high_policy,
            'skill_prior' : skill_prior, 
            'buffer' : buffer, 
            'qfs' : qfs,            
        }        

        # See rl_cfgs in LVD/configs/common.yaml 
        sac_config = {**self.rl_cfgs}

        sac_config.update(sac_modules)
        sac = SAC(sac_config).cuda()

        self.collector, self.sac = collector, sac

    def fit(self):
        # for task_obj in self.tasks:
        for task_obj in self.tasks:
            task_name = str(task_obj)
            self.prep()

            torch.save({
                "model" : self.sac,
                # "collector" : collector if env_name != "carla" else None,
                # "task" : task_obj if env_name != "carla" else np.array(task_obj),
                # "env" : env if env_name != "carla" else None,
            }, f"{self.cfg.weights_path}/{task_name}.bin")   
            
            # TODO : collector.env로 통일. ㅈㄴ헷갈림. 

            with self.collector.env.set_task(task_obj):
                state = self.collector.env.reset()
                # 이건 env에 넣을까? 
                # task = state_processor.get_goals(state)
                # print("TASK : ",  state_processor.state_goal_checker(state, env, mode = "goal") )
                # print("TASK : ",  GOAL_CHECKERS[args.env_name](   GOAL_TRANSFORM[args.env_name](state)  ))

                ewm_rwds = 0
                early_stop = 0
                for n_ep in range(self.cfg.n_episode+1):                    
                    log = self.train_policy(n_ep)
                    log['n_ep'] = n_ep
                    log[f'{task_name}_return'] = log['tr_return']
                    if 'GCSL_loss' in log.keys():
                        # 반대가 좋긴한데 zero-division error나옴. 
                        log[f'GCSL over return'] = log['tr_return'] / log['GCSL_loss'] 
                    del log['tr_return']

                    if (n_ep + 1) % self.cfg.render_period == 0:
                        log['policy_vis'] = self.visualize()

                    ewm_rwds = 0.8 * ewm_rwds + 0.2 * log[f'{task_name}_return']
                    if ewm_rwds > self.cfg.early_stop_threshold:
                        early_stop += 1
                    else:
                        early_stop = 0
                    
                    if early_stop == 10:
                        print("Converged enough. Early Stop!")
                        break

                    log = {f"{task_name}/{k}": log[k] for k in log.keys()}
                    wandb.log(log)
                    # clear plot
                    plt.cla()


    def train_policy(self, n_ep):

        log = {}

        # ------------- Collect Data ------------- #
        with self.sac.policy.expl(), self.collector.low_actor.expl() : #, collector.env.step_render():
            episode, G = self.collector.collect_episode(self.sac.policy, verbose = True)

        if np.array(episode.rewards).sum() == self.cfg.max_reward: # success 
            print("success")

        self.sac.buffer.enqueue(episode.as_high_episode()) 
        log['tr_return'] = sum(episode.rewards)

        
        if self.sac.buffer.size < self.cfg.rl_batch_size or n_ep < self.cfg.precollect:
            return log
        
        
        if n_ep == self.cfg.precollect:
            step_inputs = edict(
                episode = n_ep,
                G = G
            )
            # Q-warmup
            print("Warmup Value function")
            self.sac.warmup_Q(step_inputs)

        # n_step = self.n_step(episode)
        # print(f"Reuse!! : {n_step}")
        # for _ in range(max(n_step, 1)):
        for _ in range(self.cfg.step_per_ep):
            step_inputs = edict(
                episode = n_ep,
                G = G
            )
            stat = self.sac.update(step_inputs)

        log.update(itemize(stat))

        return log

    def visualize(self):
        
        if self.env.name == "maze":
            return draw_maze(plt.gca(), self.env, list(self.sac.buffer.episodes)[-20:])
        elif self.env.name == "kitchen":
            with self.sac.policy.expl(), self.collector.low_actor.expl() : #, collector.env.step_render():
                imgs = self.collector.collect_episode(self.sac.policy, vis = True)
            imgs = np.array(imgs).transpose(0, 3, 1, 2)
            return wandb.Video(imgs, fps=50, format = "mp4")
        else:
            NotImplementedError

    def n_step(self, episode):
        return int(self.cfg.reuse_rate * len(episode) / self.cfg.batch_size)
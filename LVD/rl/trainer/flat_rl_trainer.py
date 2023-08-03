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
# from .sac import SAC

from ..agent import Flat_GCSL

from ...modules import *
from ...contrib.simpl.torch_utils import itemize
from ...utils import *
from ...collector import GC_Flat_Collector, GC_Buffer

from simpl_reproduce.maze.maze_vis import draw_maze


class Flat_RL_Trainer:
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

        self.data = []
        # result 생성시각 구분용도 
        self.run_id = cfg.run_id #get_time()
        
        # eval_data path :  logs/[ENV_NAME]/[STRUCTURE_NAME]/[PRETRAIN_OVERRIDES]/[RL_OVERRIDES]/[TASK]
        self.result_path = self.cfg.result_path
        os.makedirs(self.result_path, exist_ok= True)

    
    def set_env(self):
        envtask_cfg = self.cfg.envtask_cfg
        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
        self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.target_tasks]

    def prep(self):
        # load skill learners and prior policy
    
        # learnable
        model = self.cfg.skill_trainer.load(self.cfg.skill_weights_path, self.cfg).model
        policy = deepcopy(model.prior_policy)
        # non-learnable
        # action_prior = Normal_Distribution(self.cfg.normal_distribution)
        # action_prior.requires_grad_(False) 

        qfs = [SequentialBuilder(self.cfg.q_function)  for _ in range(2)]
        buffer = GC_Buffer(self.cfg).to(policy.device)
        collector = GC_Flat_Collector(
            self.env,
            time_limit= self.cfg.time_limit,
        )


        agent_submodules = {
            'policy' : policy,
            # 'skill_prior' : get_fixed_dist(), 
            'buffer' : buffer, 
            'qfs' : qfs,            
        }        

        # See rl_cfgs in LVD/configs/common.yaml 
        rl_agent_config = edict({**self.cfg})
        rl_agent_config.update(agent_submodules)

        rl_agent_config['target_kl_start'] = np.log(self.cfg.action_dim).astype(np.float32)
        rl_agent_config['target_kl_end'] = np.log(self.cfg.action_dim).astype(np.float32)


        agent = Flat_GCSL(rl_agent_config).cuda()

        self.collector, self.agent = collector, agent

    def fit(self):
        for seed in self.cfg.seeds:
            self.__fit__(seed)

    def __fit__(self, seed):
        seed_everything(seed)

        # for task_obj in self.tasks:
        for task_obj in self.tasks[1:2]:
            task_name = str(task_obj)
            self.prep()

            torch.save({
                "model" : self.agent,
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

                    log, ewm_rwds = self.postprocess_log(log, task_name, n_ep, ewm_rwds)

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
        with self.sac.policy.expl() : #, collector.env.step_render():
            episode, G = self.collector.collect_episode(self.sac.policy, verbose = True)

        if np.array(episode.rewards).sum() == self.cfg.max_reward: # success 
            print("success")

        self.sac.buffer.enqueue(episode) 
        log['tr_return'] = sum(episode.rewards)

        
        if self.sac.buffer.size < self.cfg.rl_batch_size or n_ep < self.cfg.precollect:
            return log
        
        n_step = self.n_step(episode)
        # print(f"Reuse!! : {n_step}")

        for _ in range(max(n_step, 1)):
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
            with self.sac.policy.expl() : #, collector.env.step_render():
                imgs = self.collector.collect_episode(self.sac.policy, vis = True)
            imgs = np.array(imgs).transpose(0, 3, 1, 2)
            return wandb.Video(imgs, fps=50, format = "mp4")
        else:
            NotImplementedError

    def n_step(self, episode):
        return int(self.cfg.reuse_rate * len(episode) / self.cfg.batch_size)
    

    
    def postprocess_log(self, log, task_name, n_ep, ewm_rwds):

        log['n_ep'] = n_ep
        log[f'{task_name}_return'] = log['tr_return']
        if 'GCSL_loss' in log.keys():
            log[f'GCSL over return'] = log['tr_return'] / log['GCSL_loss'] 
        del log['tr_return']

        if (n_ep + 1) % self.cfg.render_period == 0:
            log['policy_vis'] = self.visualize()

        ewm_rwds = 0.8 * ewm_rwds + 0.2 * log[f'{task_name}_return']
        log = {f"{task_name}/{k}": log[k] for k in log.keys()}

        return log, ewm_rwds
    
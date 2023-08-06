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
from ..agent import SAC
from ...modules import *
from ...contrib.simpl.torch_utils import itemize
from ...utils import *
from ...collector import GC_Hierarchical_Collector, GC_Buffer

import pandas as pd
from simpl_reproduce.maze.maze_vis import draw_maze


class GC_Skill_RL_Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.set_env()
        os.makedirs(self.cfg.weights_path, exist_ok= True)
        self.rl_cfgs = { attr_nm : getattr(self.cfg, attr_nm) for attr_nm in self.cfg.rl_cfgs}
        # ------------- Logger ------------- #
        wandb.init(
            project = self.cfg.project_name,
            name = self.cfg.wandb_run_name,
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
        model = self.cfg.skill_trainer.load(self.cfg.skill_weights_path, self.cfg).model
    
        # learnable
        high_policy = deepcopy(model.prior_policy)

        # non-learnable
        low_actor = deepcopy(model.skill_decoder)
        skill_prior = deepcopy(model.prior_policy.skill_prior)
        low_actor.eval()
        low_actor.requires_grad_(False)
        skill_prior.requires_grad_(False) 

        qfs = [ SequentialBuilder(self.cfg.q_function)  for _ in range(2)]
        # buffer = GC_Buffer(self.cfg.state_dim, self.cfg.skill_dim, self.cfg.n_goal, self.cfg.buffer_size, self.env.name, model.tanh, self.cfg.hindsight_relabel).to(high_policy.device)
        
        buffer = self.cfg.buffer_cls(self.cfg).to(high_policy.device)

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
        # rl_config = {**self.rl_cfgs}
        rl_agent_config = edict({**self.cfg})
        rl_agent_config.update(sac_modules)

        agent = self.cfg.rlAgentCls(rl_agent_config).cuda()
        # sac = SAC(rl_config).cuda()

        self.collector, self.agent = collector, agent

    def fit(self):
        for seed in self.cfg.seeds:
            self.__fit__(seed)


    def __fit__(self, seed):
        seed_everything(seed)

        # for task_obj in self.tasks:
        for task_obj in self.tasks:
            task_name = f"{str(task_obj)}_seed:{seed}"
            self.prep()

            torch.save({
                "model" : self.agent,
            }, f"{self.cfg.weights_path}/{task_name}.bin")   
            
            # TODO : collector.env로 통일. ㅈㄴ헷갈림. 

            with self.collector.env.set_task(task_obj):
                self.collector.env.reset()

                ewm_rwds = 0
                early_stop = 0
                for n_ep in range(self.cfg.n_episode+1):                    
                    log = self.train_policy(n_ep, seed)

                    log, ewm_rwds = self.postprocess_log(log, task_name, n_ep, ewm_rwds)
                    wandb.log(log)
                    # clear plot
                    plt.cla()
                    
                    # 매 ep끝날 때 마다 저장. 
                    if os.path.exists(f"{self.result_path}/rawdata.csv"):
                        df = pd.read_csv(f"{self.result_path}/rawdata.csv")
                        new_data = pd.DataFrame(self.data)
                        df = pd.concat((df, new_data), axis = 0)
                        
                    else:
                        df = pd.DataFrame(self.data)
                    
                    df.drop_duplicates(inplace = True)
                    df.to_csv(f"{self.result_path}/rawdata.csv", index = False)


            torch.save({
                "model" : self.agent,
            }, f"{self.cfg.weights_path}/{task_name}.bin")   


    def train_policy(self, n_ep, seed):
        log = {}
        success = False
        # ------------- Collect Data ------------- #
        with self.agent.policy.expl(), self.collector.low_actor.expl(): #, collector.env.step_render():
            episode, G = self.collector.collect_episode(self.agent.policy, verbose = True)

        if np.array(episode.rewards).sum() == self.cfg.max_reward: # success 
            success = True
            print(len(episode.states))
            print("success")

        high_ep, high_ep_relabeled = episode.as_high_episode()
        self.agent.buffer.enqueue(high_ep) 
        # if high_ep_relabeled is not None:
        #     # nothing acheived
        #     self.agent.buffer.enqueue(high_ep_relabeled) 

        # high_ep = episode.as_high_episode()
        # self.agent.buffer.enqueue(high_ep) 
        log['tr_return'] = sum(episode.rewards)

        # ------------- Logging Data ------------- #
        data = edict(
            env = self.env.name, 
            task = str(self.collector.env.task),
            seed = seed, 
            episode = n_ep,
            reward  = sum(episode.rewards),
            success = np.array(episode.dones).sum() != 0,
            run_id = self.run_id
        )
        self.data.append(data)



        # ------------- Precollect phase ------------- #
        if self.agent.buffer.size < self.cfg.rl_batch_size or n_ep < self.cfg.precollect:
            return log
        
        # ------------- Warming up phase ------------- #
        elif n_ep == self.cfg.precollect:
            step_inputs = edict(
                episode = n_ep,
                G = G
            )
            print("Warmup Value function")
            self.agent.warmup_Q(step_inputs)
        else:
            pass 
        
        # ------------- Policy learning phase ------------- #
        n_step = self.n_step(high_ep)
        print(f"Reuse!! : {n_step}")
        for _ in range(max(n_step, 1)):
            step_inputs = edict(
                episode = n_ep,
                G = G
            )
            stat = self.agent.update(step_inputs)

        log.update(itemize(stat))

        return log

    def visualize(self):
        print("visulaize!")
        if self.env.name == "maze":
            return draw_maze(plt.gca(), self.env, list(self.agent.buffer.episodes)[-20:])
        elif self.env.name == "kitchen":
            with self.agent.policy.expl(), self.collector.low_actor.expl() : #, collector.env.step_render():
                imgs = self.collector.collect_episode(self.agent.policy, vis = True)
            imgs = np.array(imgs).transpose(0, 3, 1, 2)
            return wandb.Video(imgs, fps=50, format = "mp4")
        else:
            NotImplementedError

    def n_step(self, episode):
        return int(self.cfg.reuse_rate * len(episode) / self.cfg.rl_batch_size)
    
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
    

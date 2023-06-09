from copy import deepcopy
from LVD.collector import GC_Hierarchical_Collector
from easydict import EasyDict as edict 
import pandas as pd 
import os 
import numpy as np
import torch

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg 

        # env
        envtask_cfg = cfg.envtask_cfg
        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
        self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.target_tasks]
        
        # model 
        model = cfg.skill_trainer.load(cfg.zeroshot_weight, cfg).model
        self.high_policy = deepcopy(model.prior_policy)
        # non-learnable
        low_actor = deepcopy(model.skill_decoder)
        skill_prior = deepcopy(model.prior_policy.skill_prior)
        low_actor.eval()
        low_actor.requires_grad_(False)
        skill_prior.requires_grad_(False) 
        
        self.collector = GC_Hierarchical_Collector(
            self.env,
            low_actor,
            horizon = cfg.subseq_len -1,
            time_limit= cfg.time_limit,
        )

        self.eval_data = []

        
        self.eval_data_prefix = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/{cfg.rl_overrides}/"

        self.eval_rawdata_path = f"{self.eval_data_prefix}/{cfg.env.env_name}_{cfg.structure}_{cfg.run_name}_{cfg.rl_overrides}_eval_data_raw.csv"
        self.eval_data_path = f"{self.eval_data_prefix}/{cfg.env.env_name}_{cfg.structure}_{cfg.run_name}_{cfg.rl_overrides}_eval_data_agg.csv"

        os.makedirs(self.eval_data_prefix, exist_ok= True)

    @staticmethod
    def flat_cols(df):
        df.columns = [' / '.join(x) for x in df.columns.to_flat_index()]
        return df

    def evaluate(self):
        for seed in self.seeds:
            for task in self.tasks:
                if not self.cfg.zeroshot:
                    # load finetuned weight 
                    finetuned_model_path = f"{self.finetune_weight_prefix}/{str(task)}.bin"
                    self.high_policy = torch.load(finetuned_model_path).policy

                
                self.eval_singleTask(seed, task)
        
        df = pd.DataFrame(self.eval_data)
        df.to_csv( self.eval_rawdata_path, index = False )
 
        aggregated = df[['task', 'reward', 'success']].groupby('task', as_index= False).agg(['mean', 'std']).pipe(self.flat_cols).reset_index()
        aggregated.to_csv(self.eval_data_path, index = False)

    def eval_singleTask(self, seed, task):
        with self.collector.env.set_task(task):
            for _ in range(self.cfg.n_eval):
                with self.high_policy.expl(), self.collector.low_actor.expl() : #, collector.env.step_render():
                    episode, G = self.collector.collect_episode(self.high_policy, verbose = True)
                data = edict(
                    env = self.env.name, 
                    task = str(task),
                    seed = seed, 
                    reward  = sum(episode.rewards),
                    success = np.array(episode.dones).sum() != 0
                )
                self.eval_data.append(data)
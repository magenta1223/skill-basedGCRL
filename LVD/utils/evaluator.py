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

        os.makedirs(self.cfg.eval_data_prefix, exist_ok= True)

        self.eval_methods = edict(
            zeroshot = self.eval_zeroshot,
            finetuned = self.eval_finetuned,
            learningGraph = self.eval_learningGraph
        )

    @staticmethod
    def flat_cols(df):
        def parseCol(flatten_col):
            if flatten_col[0] == "".join(flatten_col):
                return flatten_col[0]
            else:
                return '/'.join(flatten_col)

        df.columns = [parseCol(x) for x in df.columns.to_flat_index()]
        return df


    def evaluate(self):
        assert self.cfg.eval_mode in self.eval_methods, f"Invalid evaluation methods. Valid choices are {self.eval_methods.keys()}"
        self.eval_methods[self.cfg.eval_mode]()
    
    def eval_zeroshot(self):
        for seed in self.cfg.seeds:
            for task in self.tasks:
                self.eval_singleTask(seed, task)
        
        df = pd.DataFrame(self.eval_data)
        df.to_csv( f"{self.cfg.eval_data_prefix}/zeroshot_rawdata.csv", index = False )
        aggregated = df[['task', 'reward', 'success']].groupby('task', as_index= False).agg(['mean', 'std']).pipe(self.flat_cols).reset_index()
        aggregated.to_csv(f"{self.cfg.eval_data_prefix}/zeroshot.csv", index = False)

        
        if self.env.name == "kitchen":
            # mode dropping이 훨~씬 좋다 
            easy_task = ['MKBT', 'MKBL', 'BLSH', 'MBLH', 'KTSH']
        else:
            easy_tasks = ['[24. 34.]', '[23. 14.]', '[18.  8.]']

        df['task_type'] = df['task'].apply(lambda x : 'easy' if x in easy_task else 'hard' )

        df_tasktype= df.drop(['env', 'task'], axis = 1).groupby('task_type', as_index= False).agg(['mean', 'std']).pipe(self.flat_cols).reset_index()
        df_tasktype.to_csv(f"{self.cfg.eval_data_prefix}/zeroshot_tasktype.csv", index = False)
        


    def eval_finetuned(self):
        try:
            for seed in self.cfg.seeds:
                for task in self.tasks:
                    # load finetuned weight 
                    # seed 
                    finetuned_model_path = f"{self.cfg.finetune_weight_prefix}/{str(task)}_seed:{seed}.bin"
                    print(finetuned_model_path)
                    # ckpt = torch.load(finetuned_model_path)
                    self.high_policy = torch.load(finetuned_model_path)['model'].policy
                    self.eval_singleTask(seed, task)
        except:
            for task in self.tasks:
                # load finetuned weight 
                # seed 
                finetuned_model_path = f"{self.cfg.finetune_weight_prefix}/{str(task)}.bin"
                # ckpt = torch.load(finetuned_model_path) # weight를 불러왔는데 이게 구버전이라 호환이 안됨. -> 
                _weight = torch.load(finetuned_model_path)['model'].policy.state_dict()
                self.high_policy.load_state_dict(_weight)
                self.eval_singleTask(seed, task)         
                
        df = pd.DataFrame(self.eval_data)
        df.to_csv( f"{self.cfg.eval_data_prefix}/finetune_rawdata.csv", index = False )
        aggregated = df[['task', 'reward', 'success']].groupby('task', as_index= False).agg(['mean', 'std']).pipe(self.flat_cols).reset_index()
        aggregated.to_csv(f"{self.cfg.eval_data_prefix}/finetuned.csv", index = False)

    def eval_learningGraph(self):
        # 학습과정에서 생성된 csv 파일 불러와서 
        # 슈루룩 
        print(f"Loading : {self.cfg.eval_rawdata_path}")
        raw_data = pd.read_csv(self.cfg.eval_rawdata_path)
        # 여러번 수행할 경우 run_id가 쌓임. 원치 않음. run_id를 선택할 수 있게 input 추가 
        choices = raw_data['run_id'].unique()
        if len(choices) > 1:
            run_id = input(f"Select Run ID. Choices are {choices}")
            raw_data = raw_data.loc[raw_data['run_id'] == run_id]
        
        raw_data.drop(['run_id'], axis = 1, inplace= True)

        aggregated = raw_data[['episode', 'task', 'reward']].groupby(['task', 'episode'], as_index= False).agg(['mean', 'std'])
        aggregated = aggregated.reset_index().pipe(self.flat_cols)

        
        for task, group in aggregated.groupby('task'):
            group.drop(['task'], axis = 1, inplace = True)
            group.columns = ['x', 'y', 'err']
            group['y'] = group['y'].ewm(alpha = 0.2).mean()
            group.to_csv(f"{self.cfg.eval_data_prefix}/{task}.csv", index = False)
        
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
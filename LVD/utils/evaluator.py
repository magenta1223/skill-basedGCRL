from copy import deepcopy
from LVD.collector import GC_Hierarchical_Collector, GC_Flat_Collector
from easydict import EasyDict as edict 
import pandas as pd 
import os 
import numpy as np
import torch
from .general_utils import *
import json 


def clean_colname(col_name):
    if col_name.count("/") == 1:
        # return "/".join(col_name.split("/")[:-1])
        return col_name.split("/")[0]
    else:
        return col_name


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg 

        self.task_type_order = {
            'seen' : 0,
            'small' : 1,
            'medium' : 2,
            'large' : 3
        }
        
        self.env_scoreFactor = {
            'kitchen' : 25,
            'maze' : 1,
            'toy' : 1
        }

        # env
        envtask_cfg = cfg.envtask_cfg
        self.env = envtask_cfg.env_cls(**envtask_cfg.env_cfg)
        
        self.env_name = self.env.name
        
        if cfg.eval_mode == "zeroshot":
            self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.zeroshot_tasks]
        else:
            self.tasks = [envtask_cfg.task_cls(task) for task in envtask_cfg.fewshot_tasks]

        if not cfg.eval_mode == "rearrange":            
            # model 
            try:
                model = cfg.skill_trainer.load(cfg.zeroshot_weight, cfg).model
            except:
                print(f"{cfg.zeroshot_weight} No end.")
                model = cfg.skill_trainer.load(cfg.zeroshot_weight_before_rollout, cfg).model

            if "flat" in cfg.structure:
                self.policy = deepcopy(model.prior_policy)
                # non-learnable
                
                self.collector = GC_Flat_Collector(
                    self.env,
                    time_limit= cfg.time_limit,
                )

            else:
                self.high_policy = deepcopy(model.prior_policy)
                # non-learnable
                low_actor = deepcopy(model.skill_decoder)
                skill_prior = deepcopy(model.prior_policy.skill_prior)
                low_actor.eval()
                low_actor.requires_grad_(False)
                skill_prior.requires_grad_(False) 
                
                self.collector = GC_Hierarchical_Collector(
                    cfg,
                    self.env,
                    low_actor,
                    horizon = cfg.subseq_len -1,
                    time_limit= cfg.time_limit,
                )

        self.eval_data = []

        os.makedirs(self.cfg.eval_data_prefix, exist_ok= True)

        self.eval_methods = edict(
            zeroshot = self.eval_zeroshot,
            finetune = self.eval_finetuned,
            learningGraph = self.eval_learningGraph,
            rearrange = self.rearrange_taskgroup
        )

        self.set_logger()

    def set_logger(self):
        self.run_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}/{self.cfg.run_name}"
        
        log_path = f"{self.run_path}/{self.cfg.job_name}.log"
        self.logger = Logger(log_path, verbose= True)


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
    
    def task_mapping(self, df, env_name = None):
        with open(f"./assets/{self.env_name}_tasks.json") as f:
            task_dict = json.load(f)
        df['task_type'] = df['task'].map(task_dict)
        return df 
        
    def aggregate(self, df = None):
        
        if df is None:
            # raw data
            df = pd.DataFrame(self.eval_data)
            df.to_csv( f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_rawdata.csv", index = False )

        df['reward'] = df['reward'].apply(lambda x: x * self.env_scoreFactor[self.env_name])

        per_task_target_cols = ['seed', 'task', 'reward', 'success']
        per_task_groupby = ['seed', 'task']
        tasktype_groupby = ['seed', 'task_type']
        sort_cols = ['order']
        pertask_target_cols = [ 'task', 'reward', 'success']
        tasktype_target_cols = ['task_type', 'reward', 'success']

        if self.cfg.eval_mode =="finetune":
            per_task_target_cols.append('shot')
            per_task_groupby.append('shot')
            tasktype_groupby.append('shot')
            sort_cols.append('shot')
            pertask_target_cols = ['task', 'shot', 'reward', 'success']
            tasktype_target_cols = ['task_type', 'shot', 'reward', 'success']

        # aggregate along  task, seed 
        aggregated = df[per_task_target_cols].groupby(per_task_groupby, as_index= False).agg(['mean']).pipe(self.flat_cols).reset_index()
    
        aggregated.columns = [clean_colname(col) for col in aggregated.columns]
        # seed 제외 
        per_task_groupby.remove("seed")
        aggregated = df[per_task_target_cols].groupby(per_task_groupby, as_index= False).agg(['mean', 'sem']).pipe(self.flat_cols).reset_index()

    
        aggregated['reward'] = aggregated.apply(lambda row: f"{round(row['reward/mean'], 1)} \\pm {round(row['reward/sem'] * 1.96, 1)}", axis = 1)
        aggregated['success'] = aggregated.apply(lambda row: f"{round(row['success/mean'], 1)} \\pm {round(row['success/sem'] * 1.96, 1)}", axis = 1)

        aggregated = aggregated[pertask_target_cols]

        aggregated.to_csv(f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}.csv", index = False)
        
        self.logger.log(f"Done : {self.cfg.eval_data_prefix}/{self.cfg.eval_mode}.csv")


        # --------------------------- # 

        df = self.task_mapping(df)

        unseen_rwd_mu, unseen_rwd_ste = df.loc[df['task_type'] != "seen"][['reward']].agg(['mean', 'sem']).values
        unseen_scs_mu, unseen_scs_ste = df.loc[df['task_type'] != "seen"][['success']].agg(['mean', 'sem']).values

        unseen_avg = {
            'task_type' : 'unseen Avg.',
            'reward/mean' : unseen_rwd_mu,
            'reward/sem' : unseen_rwd_ste,
            'success/mean' : unseen_scs_mu,
            'success/sem' : unseen_scs_ste,
        }

        # aggregate per task group            
        df_tasktype= df.drop(['env', 'task'], axis = 1).groupby(tasktype_groupby, as_index= False).agg(['mean']).pipe(self.flat_cols).reset_index()
        
        df_tasktype.columns = [clean_colname(col) for col in df_tasktype.columns]

        tasktype_groupby.remove("seed")
        df_tasktype= df_tasktype.groupby(tasktype_groupby, as_index= False).agg(['mean', 'sem']).pipe(self.flat_cols).reset_index()
        
        
        df_tasktype['order'] = df_tasktype['task_type'].map(self.task_type_order)
        df_tasktype = df_tasktype.sort_values(by = sort_cols)
        df_tasktype = pd.concat(( df_tasktype, pd.DataFrame(unseen_avg) ), axis = 0).reset_index(drop=True).drop(['order'], axis = 1)

        
        df_tasktype['reward'] = df_tasktype.apply(lambda row: f"{round(row['reward/mean'], 1)} \\pm {round(row['reward/sem'] * 1.96, 1)}", axis = 1)
        df_tasktype['success'] = df_tasktype.apply(lambda row: f"{round(row['success/mean'], 1)} \\pm {round(row['success/sem'] * 1.96, 1)}", axis = 1)
        df_tasktype = df_tasktype[tasktype_target_cols]
        
        
        df_tasktype.to_csv(f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_tasktype.csv", index = False)
        self.logger.log(f"Done : {self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_tasktype.csv")
        
        
        
        
        
        
        
        
    def eval_zeroshot(self):
        
        df = None
        zeroshot_rawdata_path = f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_rawdata.csv"
        
        if os.path.exists(zeroshot_rawdata_path):
            df = pd.read_csv(zeroshot_rawdata_path)
        
        for seed in self.cfg.seeds:
            for task in self.tasks:
                if df is not None:
                    # previously evaluated with same seed, task, shot 
                    if df.loc[(df['task'] == str(task)) & (df['seed'] == seed) ].shape[0] == self.cfg.n_eval:
                        self.logger.log(f"Skip :  {task}_seed:{seed} is previoulsy evaluated.")
                        continue
                self.eval_singleTask(seed, task)
        
        if df is not None:
            # raw data
            eval_df = pd.DataFrame(self.eval_data)
            df = pd.concat((df, eval_df), axis = 0)
            df.to_csv( zeroshot_rawdata_path, index = False )
            
        
        
        
        self.aggregate(df)
        
    def eval_finetuned(self):
        shots = self.cfg.shots
        # early_stop 
        
        df = None
        finetune_rawdata_path = f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_rawdata.csv"
        
        if os.path.exists(finetune_rawdata_path):
            df = pd.read_csv(finetune_rawdata_path)
        
        for seed in self.cfg.seeds:
            for task in self.tasks:
                # load finetuned weight 
                # seed
                earlystop_model_path = f"{self.cfg.finetune_weight_prefix}/{str(task)}_seed:{seed}.bin"
                for shot in shots: 
                    if df is not None:
                        # previously evaluated with same seed, task, shot 
                        if df.loc[(df['task'] == str(task)) & (df['shot'] == shot) & (df['seed'] == seed) ].shape[0] == self.cfg.n_eval:
                            self.logger.log(f"Skip :  {task}_shot:{shot}_seed:{seed} is previoulsy evaluated.")
                            continue

                    # {task}_seed:{seeed}_ep{shot}.bin
                    # 여기서 early stop이 적용된 경우 shot이 없을 수 있음. 
                    # early stop이 적용되면 -> {task}_seed:{seeed}.bin 을 사용
                    # early stop 적용 여부는 -> shot path가 없을 때 
                    finetuned_model_path = f"{self.cfg.finetune_weight_prefix}/{str(task)}_seed:{seed}_ep{shot}.bin"
                    if os.path.exists(finetuned_model_path):
                        self.logger.log(f"Evaluating : {finetuned_model_path}")
                        self.high_policy = torch.load(finetuned_model_path)['model'].policy
                    else:
                        self.logger.log(f"Early Stopped, Evaluating : {earlystop_model_path}")
                        self.high_policy = torch.load(earlystop_model_path)['model'].policy
                        
                    # self.high_policy = torch.load(finetuned_model_path)['model'].policy
                    self.eval_singleTask(seed, task, shot)
                
        if df is not None:
            # raw data
            eval_df = pd.DataFrame(self.eval_data)
            df = pd.concat((df, eval_df), axis = 0)
            df.to_csv( f"{self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_rawdata.csv", index = False )
            
        
        self.aggregate(df)

    def eval_learningGraph(self):
        # 학습과정에서 생성된 csv 파일 불러와서 
        # 슈루룩 
        
        self.logger.log("Starting Evaluation : Learning Graph")
        self.logger.log(f"Loading : {self.cfg.eval_rawdata_path}")

        raw_data = pd.read_csv(self.cfg.eval_rawdata_path)
        # 여러번 수행할 경우 run_id가 쌓임. 원치 않음. run_id를 선택할 수 있게 input 추가 
        choices = raw_data['run_id'].unique()
        if len(choices) > 1:
            run_id = input(f"Select Run ID. Choices are {choices}")
            raw_data = raw_data.loc[raw_data['run_id'] == run_id]
        
        raw_data.drop(['run_id'], axis = 1, inplace= True)

        aggregated = raw_data[['episode', 'task', 'reward']].groupby(['task', 'episode'], as_index= False).agg(['mean', 'sem'])
        aggregated = aggregated.reset_index().pipe(self.flat_cols)

        
        for task, group in aggregated.groupby('task'):
            group.drop(['task'], axis = 1, inplace = True)
            group.columns = ['x', 'y', 'err']
            group['y'] = group['y'].ewm(alpha = 0.2).mean()
            group.to_csv(f"{self.cfg.eval_data_prefix}/{task}.csv", index = False)
        
            self.logger.log(f"Done : {self.cfg.eval_data_prefix}/{task}.csv")


    def eval_singleTask(self, seed, task, shot = None):
        """
        seed : used for few-shot adaptation
        task : target goal
        shot : n-shot for logging. optional.
        """
        
        
        if "flat" in self.cfg.structure:
            with self.collector.env.set_task(task):
                for _ in range(self.cfg.n_eval):
                    with self.policy.no_expl(): #, collector.env.step_render():
                        episode, G = self.collector.collect_episode(self.policy, verbose = True)
                    data = edict(
                        env = self.env.name, 
                        task = str(task),
                        seed = seed, 
                        reward  = sum(episode.rewards),
                        success = np.array(episode.dones).sum() != 0
                    )
                    if shot is not None:
                        data['shot'] = shot
                    self.eval_data.append(data)


        else:
            with self.collector.env.set_task(task):
                for _ in range(self.cfg.n_eval):
                    with self.high_policy.no_expl(), self.collector.low_actor.no_expl() : #, collector.env.step_render():
                        episode, G = self.collector.collect_episode(self.high_policy, verbose = True)
                    data = edict(
                        env = self.env.name, 
                        task = str(task),
                        seed = seed, 
                        reward  = sum(episode.rewards),
                        success = np.array(episode.dones).sum() != 0
                    )
                    if shot is not None:
                        data['shot'] = shot
                    self.eval_data.append(data)
                    
                    
    def rearrange_taskgroup(self):
        # 현재 dataset 내부의 모든 logged data에 대해 다음과 같이 진행. 
        # 1. [EVAL_METHODS]_tasktype.csv 가 존재하는 모든 폴더를 찾는다. 
        # 2. 해당 폴더 안에는 [EVAL_METHODS]_rawdata.csv가 존재한다. 이를 다시 task_group에 맞춰 재작성 
        # 3. task group은 asset에서 불러오기 
        
        
        self.logger.log("Starting Evaluation : Task Rearrnagement")
        

        # 현재 디렉토리부터 시작하여 모든 하위 디렉토리를 검색
        target_folders = []

        for root, dirs, files in os.walk('.'):
            for file in files:
                folder_path = os.path.abspath(root)
                if file.endswith('zeroshot_tasktype.csv'):
                    # print(f"폴더 경로: {folder_path}, 파일명: {file}")
                    target_folders.append(folder_path)
        
        for folder_path in target_folders:
            rawdata_path = f"{folder_path}/zeroshot_rawdata.csv"
            if not os.path.exists(rawdata_path):
                # print(f"{folder_path} does not have rawdata")
                self.logger.log(f"{folder_path} does not have rawdata")
                continue
            
            if "maze" in rawdata_path:
                env_name = "maze"
            else:
                env_name = "kitchen"
            
            # zeroshot !! 
            df = pd.read_csv(rawdata_path)
            
            self.cfg.eval_data_prefix = folder_path 
            self.cfg.eval_mode = "zeroshot"
            self.env_name = env_name


            # self.aggregate(df)
            try:
                self.aggregate(df)
            except:
                self.logger.log(f"Failed : {self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_tasktype.csv")
                
        target_folders = []
        
        for root, dirs, files in os.walk('.'):
            for file in files:
                folder_path = os.path.abspath(root)
                if file.endswith('finetune_tasktype.csv'):
                    target_folders.append(folder_path)
        
        for folder_path in target_folders:
            rawdata_path = f"{folder_path}/finetune_rawdata.csv"
            if not os.path.exists(rawdata_path):                
                self.logger.log(f"{folder_path} does not have rawdata")
                continue
        
            if "maze" in rawdata_path:
                env_name = "maze"
            else:
                env_name = "kitchen"
        
            df = pd.read_csv(rawdata_path)
            
            self.cfg.eval_data_prefix = folder_path 
            self.cfg.eval_mode = "finetune"
            self.env_name = env_name

            # self.aggregate(df)

            
            try:
                self.aggregate(df)
            except:
                self.logger.log(f"Failed : {self.cfg.eval_data_prefix}/{self.cfg.eval_mode}_tasktype.csv")

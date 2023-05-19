import os
import time
from datetime import datetime
from glob import glob
import torch
import wandb
from tqdm import tqdm
from math import sqrt
import numpy as np

from easydict import EasyDict as edict
from copy import deepcopy

from LVD.utils import *
# from ..configs.data.kitchen import KitchenEnvConfig

from LVD.configs.env import *
# from LVD.configs.model import *
import hydra

from omegaconf import DictConfig

from LVD.models import *
import warnings

warnings.filterwarnings("ignore")



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Logger:
    def __init__(self, log_path, verbose = True):
        self.log_path =log_path
        self.verbose = verbose
        

    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


    def loss_dict_log(self, loss_dict, set_name):

        message = set_name.upper() + " "
        for k, v in loss_dict.items():
            message += f"{k.replace(set_name.upper() + '_', '')} : {v:.5f} "

        return message


class BaseTrainer:

    def __init__(self, cfg):
        
        self._cfg = cfg
        self.cfg = edict(self.get_class_cfg(cfg))

        # for k, v in self.cfg.env.items():
            # self.cfg[k] = v
        
        self.cfg.update(self.cfg.env)
        self.cfg.update(self.cfg.model)

        del self.cfg.env
        del self.cfg.model

    
        
        self.get_loader()
        self.build()
        self.prep()


        # self.model = model
        # self.prep(config)


    def get_class_cfg(self, cfg):
        """
        1. value가 dict면 -> 다 까보기
        2. value가 dict가 아니면
            - class로 만들 수 있나? -> cls로
            - 아니면 -> value로 
        3. cls로 다 바꾼 후 원래 구조의 dictioanry로 변경! 
        """
        
        new_cfg = dict()
        
        for k, v in cfg.items():
            if isinstance(v, DictConfig):
                v = self.get_class_cfg(v)
            elif isinstance(v, str) and "." in v and "/" not in v and v != ".":
                v = hydra.utils.get_class(v)
            else:
                pass
        
            new_cfg[k] = v

        return new_cfg


    def build(self):
        self.model = MODELS[self.cfg.structure](self.cfg).cuda()
        
        

    def prep(self):
        """
        Prepare for fitting
        """
        print(f"--------------------------------{self.model.structure}--------------------------------")
        self.schedulers = {
            k : self.cfg.schedulerClass(v['optimizer'], **self.cfg.scheduler_params, module_name = k) for k, v in self.model.optimizers.items()
        }

        self.schedulers_metric = {
            k : v['metric'] for k, v in self.model.optimizers.items()
        }

        print(self.schedulers)
        print(self.schedulers_metric)


        self.early_stop = 0
        self.best_summary_loss = 10**5

        # path
        self.model_path = f"{self.cfg.save_path}/{self.cfg.env_name}/{self.cfg.structure}" #os.path.join(config.save_path[0], 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.set_logger()
        self.meters = {}

    def meter_initialize(self):
        self.meters = {}


    def set_logger(self):
        logs_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}"
        os.makedirs(logs_path, exist_ok = True)

        logs = os.listdir(logs_path)

        if not logs:
            logs = [-1, 0]
            log_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}/log0.txt" # naming 꼬라지..
        else:
            logs = sorted([ int(f.replace(".txt", "").replace("log", "")) for f in logs])
            log_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}/log{max(logs) + 1}.txt"

        self.logger = Logger(log_path, verbose= False)

        config_text = "#" + "-" * 20 + "#"

        for k, v in self.cfg.items():
            config_text += f"{k} : {v}\n"

        config_text += "#" + "-" * 20 + "#"
        self.logger.log(config_text)
        print(f"Log file : ", log_path)

        self.model_id = f"log{max(logs) + 1}"


    def loop_indicator(self, e, val_loss):

        if self.best_summary_loss > val_loss:
            # record best 
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e
            # save
            if e > self.cfg.save_ckpt:
                self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
                for path in sorted(glob(f'{self.model_path}/{self.model_id}_*epoch.bin'))[:-1]:
                    os.remove(path)
        else:
            # if e > self.save_ckpt:
            #     self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
            # early self.
            self.early_stop += 1



        if self.early_stop == self.cfg.early_stop_rounds:
            return True
        else: 
            return False

    def fit(self):
        print("optimizing")
        print(f"weights save path : {self.cfg.save_path}")

        for e in range(self.cfg.epochs):
            start = time.time()

            train_loss_dict  = self.train_one_epoch(self.train_loader, e)
            valid_loss_dict = self.validate(self.val_loader, e)

            message = f'[Epoch {e}]\n'
            message += self.logger.loss_dict_log(train_loss_dict, 'train')
            message += "\n"
            message += self.logger.loss_dict_log(valid_loss_dict, 'valid')
            message += "\n"
            if self.test_loader is not None:
                test_loss_dict = self.validate(self.test_loader, e)
                message += self.logger.loss_dict_log(test_loss_dict, 'test')
                message += "\n"
            message += f'Time : {time.time() - start:.5f} s \n'

            self.logger.log(message)
            
            # skill enc, dec를 제외한 모듈은 skill에 dependent함
            # 따라서 skill이 충분히 학습된 이후에 step down을 적용해야 함. 
            if "skill_enc_dec" in self.schedulers.keys():
                skill_scheduler = self.schedulers['skill_enc_dec']
                skill_scheduler.step(valid_loss_dict[self.schedulers_metric['skill_enc_dec']])

            if e >= self.cfg.warmup_steps:
                # if self.scheduler:
                #     self.scheduler.step(valid_loss_dict['metric'])

                for module_name, scheduler in self.schedulers.items():
                    if module_name != "skill_enc_dec":
                        scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])

                if self.loop_indicator(e, valid_loss_dict['metric']):
                    print("early stop", 'loss',  valid_loss_dict['metric'])
                    break

            if e > self.cfg.save_ckpt:
                self.save(f'{self.model_path}/{self.model_id}_{e}.bin')

        self.save(f'{self.model_path}/{self.model_id}_end.bin')
        self.logger.log('Finished')           
         


    def train_one_epoch(self, loader, e):

        self.meter_initialize()


        for i, batch in enumerate(loader):
            self.model.train()
            loss = self.model.optimize(batch, e)
        

            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    self.meters[key] = AverageMeter()
  
            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])

        return { k : v.avg for k, v in self.meters.items() }

    @torch.no_grad()
    def validate(self, loader, e):
        self.model.eval()
        self.meter_initialize()

        for i, batch in enumerate(loader):
            loss = self.model.validate(batch, e)

            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    self.meters[key] = AverageMeter()
            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])


    
        return { k : v.avg for k, v in self.meters.items()}



    def save(self, path):
        self.model.eval()




        torch.save({
        "only_weights" : False,
        'model': self.model,
        # 'optimizer' : self.model.optimizer,
        # 'optim' : self.model.optim,
        # 'subgoal_optim' : self.model.subgoal_optim,
        # 'scheduler' : self.scheduler,
        # 'scaler' : self.model.scaler,
        'configuration' : self.cfg,
        'best_summary_loss': self.best_summary_loss
        }, path)


    def load(self, path):
        checkpoint = torch.load(path)
        if checkpoint['only_weights']:
            self.model.load_state_dict(checkpoint['model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            # self.model.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.model = checkpoint['model']
            self.model.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            # self.model.scaler = checkpoint['scaler']

        config = checkpoint['configuration']
        self.prep(config)
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.model.eval()

    def set_lr(self, optimizer, lr = False, ratio = False):
        if ratio and not lr:
            self.lr *= ratio
            for param_group in self.model.optimizers[optimizer]['optimizer'].param_groups:
                param_group['lr'] = self.lr
        else:
            self.lr = lr
            for _, param_group in self.model.optimizers[optimizer]['optimizer'].param_groups:
                param_group['lr'] = self.lr

        print(f"{optimizer}'s lr : {self.lr}")

    



    def get_loader(self):
        train_dataset = self.cfg.dataset_cls(self.cfg, "train")
        val_dataset = self.cfg.dataset_cls(self.cfg, "val")
        self.train_loader = train_dataset.get_data_loader(self.cfg.batch_size, num_workers = self.cfg.workers)
        self.val_loader = val_dataset.get_data_loader(self.cfg.batch_size, num_workers = self.cfg.workers)
        self.test_loader = None
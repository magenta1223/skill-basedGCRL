import os
import time
from glob import glob
import torch
from LVD.utils import *
from LVD.models import *
import warnings
import cv2
from copy import deepcopy
import numpy as np
from d4rl.pointmaze.maze_model import WALL
from matplotlib import pyplot as plt 
from easydict import EasyDict as edict   
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.get_loader()
        self.build()
        self.prep()
    
    # ---------------------- Initialize ---------------------- # 
    def build(self):
        self.model = MODELS[self.cfg.structure](self.cfg).cuda()
        
    def prep(self):
        """
        Prepare for fitting
        """

        self.schedulers_no_warmup = {
            k : self.cfg.schedulerClass(v['optimizer'], **self.cfg.scheduler_params, module_name = k) for k, v in self.model.optimizers.items() if "wo_warmup" in v
        }

        self.schedulers_warmup = {
            k : self.cfg.schedulerClass(v['optimizer'], **self.cfg.scheduler_params, module_name = k) for k, v in self.model.optimizers.items() if "wo_warmup" not in v
        }

        self.schedulers_metric = {
            k : v['metric'] for k, v in self.model.optimizers.items()
        }

        self.e = 0
        self.early_stop = 0
        self.best_summary_loss = 10**5

        self.set_logger()
        self.meters = {}

    def get_loader(self):
        train_dataset = self.cfg.dataset_cls(self.cfg, "train")
        val_dataset = self.cfg.dataset_cls(self.cfg, "val")
        self.train_loader = train_dataset.get_data_loader(self.cfg.batch_size, num_workers = self.cfg.workers)
        self.val_loader = val_dataset.get_data_loader(self.cfg.batch_size, num_workers = self.cfg.workers)
        self.test_loader = None
        
    # ----------------------------------------------------- # 

    # ---------------------- Resume ---------------------- # 
    def resume(self):
        resumed = False
        if self.cfg.resume:
            if self.cfg.resume_ckpt == "latest":
                ckpts = sorted(glob(f"{self.model_id}/"))
                if len(ckpts) == 0:
                    self.logger.log("No checkpoints")
                else:
                    try:
                        path = ckpts[-1]
                        self.logger.log(f"Loading latest checkpoints {ckpts[-1]}")
                        self = self.load(path = ckpts[-1])    
                        resumed = True
                    except:
                        self.logger.log(f"Loading failed. Default configuration or algorithm may be altered")

            elif os.path.exists(f"{self.model_id}/{self.cfg.resume_ckpt}.bin"):
                try:
                    path = f"{self.model_id}/{self.cfg.resume_ckpt}.bin"
                    self.logger.log(f"Loading {path}")
                    self = self.load(path = path)    
                    resumed = True
                except:
                    self.logger.log(f"Loading failed. Default configuration or algorithm may be altered")

            else:
                self.logger.log("Checkpoint does not exist")


        if resumed:
            self.logger.log(f"Resumed from {path}")

        if resumed and self.e >= self.cfg.mixin_start - 1:
            self.logger.log(f"Filling offline buffer")
            rollout = True 
            self.train_one_epoch(self.train_loader, self.e, rollout)
            start = deepcopy(self.e) 
        else:
            start = 0

        return self, start

    # ---------------------- Meters and Logger ---------------------- # 

    def meter_initialize(self):
        self.meters = {}

    def set_logger(self):
        self.run_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}/{self.cfg.run_name}"
        
        log_path = f"{self.run_path}/{self.cfg.job_name}.log"
        self.logger = Logger(log_path, verbose= False)

        self.model_id = f"weights/{self.cfg.env_name}/{self.cfg.structure}/{self.cfg.run_name}/skill"
        os.makedirs(self.model_id, exist_ok = True)
        os.makedirs(f"{self.run_path}/imgs", exist_ok= True)
        os.makedirs(f"{self.run_path}/data", exist_ok= True)
    
    @staticmethod
    def loss_dict_log(loss_dict, set_name):
        message = set_name.upper() + " "
        for k, v in loss_dict.items():
            message += f"{k.replace(set_name.upper() + '_', '')} : {v:.5f} "
        return message


    # ---------------------- Scheduling ---------------------- # 
    def loop_indicator(self, e, val_loss):
        if self.best_summary_loss > val_loss:
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e

        else:
            self.early_stop += 1

        if self.early_stop == self.cfg.early_stop_rounds:
            return True
        else: 
            return False
        
    def fit(self):
        print("optimizing")
        print(f"log path : {self.run_path}/train.log")
        print(f"weights path : {self.model_id}")

        self, start = self.resume()

        self.save(f'{self.model_id}/start.bin')

        for e in range(start, self.cfg.epochs):
            self.e = e 
            start = time.time()

            train_loss_dict  = self.train_one_epoch(e)
            valid_loss_dict = self.validate(self.val_loader, e)

            message = f'[Epoch {e}]\n'
            message += self.loss_dict_log(train_loss_dict, 'train')
            message += "\n"
            message += self.loss_dict_log(valid_loss_dict, 'valid')
            message += "\n"
            if self.test_loader is not None:
                test_loss_dict = self.validate(self.test_loader, e)
                message += self.loss_dict_log(test_loss_dict, 'test')
                message += "\n"
            message += f'Time : {time.time() - start:.5f} s \n'

            self.logger.log(message)
            
            for module_name, scheduler in self.schedulers_no_warmup.items():
                try:
                    msgs = scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])
                    self.logger.log(msgs)
                except:
                    pass 


            if e >= self.cfg.warmup_steps:
                for module_name, scheduler in self.schedulers_warmup.items():
                    try:
                        msgs = scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])
                        self.logger.log(msgs)
                    except:
                        pass 

                if self.loop_indicator(e, valid_loss_dict['metric']):
                    print("early stop", 'loss',  valid_loss_dict['metric'])
                    break


            if (e + 1) % 10 == 0:
                self.save(f'{self.model_id}/{e}.bin')
            
            

        self.save(f'{self.model_id}/end.bin')
        self.logger.log('Finished')           
         

    def train_one_epoch(self, e):
        self.meter_initialize()
        self.pre_epoch_hook(e)

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.pre_iter_hook()
            self.loss = self.model.optimize(batch, e)
            self.post_iter_hook()

            if not len(self.meters):
                for key in self.loss.keys():
                    self.meters[key] = AverageMeter()
  
            for k, v in self.loss.items():
                if k not in self.meters:
                    self.meters[k] = AverageMeter()
                self.meters[k].update(v, batch['states'].shape[0])
    
        self.post_epoch_hook(e)

        return { k : v.avg for k, v in self.meters.items() }

    @torch.no_grad()
    def validate(self, loader, e):
        self.model.eval()
        self.meter_initialize()
        self.pre_epoch_hook(e, validate = True)

        for i, batch in enumerate(loader):
            self.model.eval()
            self.pre_iter_hook(validate = True)
            self.loss = self.model.validate(batch, e)
            self.post_iter_hook(validate = True)

            if not len(self.meters):
                for key in self.loss.keys():
                    self.meters[key] = AverageMeter()
            for k, v in self.loss.items():
                self.meters[k].update(v, batch['states'].shape[0])
   

        self.post_epoch_hook(e, validate = True)

        return { k : v.avg for k, v in self.meters.items()}

    def save_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizers' : { module_name : optim['optimizer'].state_dict()  for module_name, optim in self.model.optimizers.items()},
            'schedulers_wo_warmup' : { module_name : scheduler.state_dict()  for module_name, scheduler in self.schedulers_no_warmup.items()},
            'schedulers_warmup' : { module_name : scheduler.state_dict()  for module_name, scheduler in self.schedulers_warmup.items()},
            'configuration' : self.cfg,
            'best_summary_loss': self.best_summary_loss,
            'epoch' : self.e
        }
    
    # ---------------------- I/O ---------------------- # 

    def save(self, path):
        self.model.eval()
        save_dict = self.save_dict()
        save_dict['cls'] = BaseTrainer
        torch.save(save_dict, path)

    def load(path, cfg = None):
        checkpoint = torch.load(path)
        if cfg is not None:
            self = checkpoint['cls'](cfg)            
        else:
            self = checkpoint['cls'](checkpoint['configuration'])            

        self.model.load_state_dict(checkpoint['model'])
        [ optim['optimizer'].load_state_dict(checkpoint['optimizers'][module_name] )  for module_name, optim in self.model.optimizers.items()]
        
        try:
            [ scheduler.load_state_dict(checkpoint['schedulers_wo_warmup'][module_name] )  for module_name, scheduler in self.schedulers_no_warmup.items()]
            [ scheduler.load_state_dict(checkpoint['schedulers_warmup'][module_name] )  for module_name, scheduler in self.schedulers_warmup.items()]
        except:
            pass 
        self.best_summary_loss = checkpoint['best_summary_loss']
        
        try:
            self.e = checkpoint['epoch']
        except:
            self.e = 0

        self.model.eval()

        return self
    
    # hooks 
    def pre_epoch_hook(self, e, validate = False):
        pass
    
    def post_epoch_hook(self, e, validate = False):
        pass

    def pre_iter_hook(self, validate = False):
        pass
    
    def post_iter_hook(self, validate = False):
        pass


class Diversity_Trainer(BaseTrainer): 

    def pre_epoch_hook(self, e, validate = False):

        if not validate:
            self.imgs = None
            self.model.render = False #
            self.model.prev_skill_encoder = deepcopy(self.model.skill_encoder)
            self.model.prev_goal_encoder = deepcopy(self.model.goal_encoder)
            
            if e > self.cfg.mixin_start:
                self.model.update_rollout_H()
            print(f"EPOCH : {e} mixin ratio : {self.train_loader.dataset.mixin_ratio} rollout length : {self.model.plan_H} discount : {self.train_loader.dataset.discount_raw}")
        else:
            pass 



    def post_epoch_hook(self, e, validate = False):
        if not validate:
            if self.imgs is not None:
                if self.cfg.env_name == "kitchen":
                    path = f"{self.run_path}/imgs/{e}.mp4"
                    print(f"Rendered : {path}")
                    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (400,400))
                    for i in range(len(self.imgs)):
                        # writing to a image array
                        out.write(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB))
                    out.release() 
                elif self.cfg.env_name == "maze":
                    path = f"{self.run_path}/imgs/{e}.png"
                    # path = f"{self.run_path}/imgs/{e}_seqIndex:{self.imgs['seq_index']}.png"
                    print(f"Rendered : {path}")
                    # self.imgs.savefig(path, bbox_inches="tight", pad_inches = 0)
                    self.imgs['fig'].savefig(path, bbox_inches="tight", pad_inches = 0)
            if e == self.cfg.mixin_start:
                self.train_loader.set_mode("with_buffer")
            self.train_loader.update_buffer()
            
            if e + 1 >= self.cfg.mixin_start:
                self.train_loader.update_ratio()
    
            if e + 1 >= self.cfg.mixin_start:
                self.model.do_rollout = True

            if e + 1 == self.cfg.mixin_start:
                self.save(f'{self.model_id}/orig_skill.bin')

            # if e > self.cfg.save_ckpt:
            #     self.save(f'{self.model_id}/{e}.bin')

        else:
            pass

    def pre_iter_hook(self, validate=False):
        if not validate:
            if len(self.train_loader.dataset.prev_buffer) >= self.cfg.offline_buffer_size:
                self.model.do_rollout = False
        else:
            pass 



    def post_iter_hook(self, validate = False):
        if not validate:
            if "states_novel" in self.loss.keys():
                self.train_loader.enqueue(self.loss.pop("states_novel"), self.loss.pop("actions_novel"), self.loss.pop("c"), self.loss.pop("seq_indices"))
            
            if "render" in self.loss.keys():
                self.imgs = self.loss.pop("render")
        else:
            if "states_novel" in self.loss.keys():
                self.loss.pop("states_novel")
                self.loss.pop("actions_novel")
                self.loss.pop("seq_indices")
                self.loss.pop("c")
            if "render" in self.loss.keys():
                self.loss.pop("render")


    def save(self, path):
        self.model.eval()
        save_dict = self.save_dict()
        save_dict['cls'] = Diversity_Trainer
        torch.save(save_dict, path)
        
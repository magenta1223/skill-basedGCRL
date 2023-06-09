import os
import time
from glob import glob
import torch
from LVD.utils import *
from LVD.models import *
import warnings
import cv2
from copy import deepcopy

warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Logger:
    def __init__(self, log_path, verbose = True):
        self.log_path =log_path
        self.verbose = verbose
        

    def log(self, message):
        if message is not None:
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
        self.cfg = cfg

        self.get_loader()
        self.build()
        self.prep()
    
    def build(self):
        self.model = MODELS[self.cfg.structure](self.cfg).cuda()
        # self.model = torch.compile(model, mode = "max-autotune")
        
            
    def prep(self):
        """
        Prepare for fitting
        """
        # print(f"--------------------------------{self.model.structure}--------------------------------")
        self.schedulers = {
            k : self.cfg.schedulerClass(v['optimizer'], **self.cfg.scheduler_params, module_name = k) for k, v in self.model.optimizers.items()
        }

        self.schedulers_metric = {
            k : v['metric'] for k, v in self.model.optimizers.items()
        }

        # print(self.schedulers)
        # print(self.schedulers_metric)
        self.e = 0
        self.early_stop = 0
        self.best_summary_loss = 10**5

        self.set_logger()
        self.meters = {}

    def set_logger(self):
        self.run_path = f"logs/{self.cfg.env_name}/{self.cfg.structure}/{self.cfg.run_name}"
        
        log_path = f"{self.run_path}/{self.cfg.job_name}.log"
        self.logger = Logger(log_path, verbose= False)

        self.model_id = f"weights/{self.cfg.env_name}/{self.cfg.structure}/{self.cfg.run_name}/skill"
        os.makedirs(self.model_id, exist_ok = True)

        os.makedirs(f"{self.run_path}/imgs", exist_ok= True)

    def meter_initialize(self):
        self.meters = {}
    
    @staticmethod
    def loss_dict_log(loss_dict, set_name):
        message = set_name.upper() + " "
        for k, v in loss_dict.items():
            message += f"{k.replace(set_name.upper() + '_', '')} : {v:.5f} "
        return message

    def loop_indicator(self, e, val_loss):

        if self.best_summary_loss > val_loss:
            # record best 
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e
            # save
            if e > self.cfg.save_ckpt:
                self.save(f'{self.model_id}/{e}.bin')
                for path in sorted(glob(f'{self.model_id}/*epoch.bin'))[:-1]:
                    os.remove(path)
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

        for e in range(self.cfg.epochs):
            self.e = e 
            start = time.time()

            train_loss_dict  = self.train_one_epoch(self.train_loader, e)
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
            
            # skill enc, dec를 제외한 모듈은 skill에 dependent함
            # 따라서 skill이 충분히 학습된 이후에 step down을 적용해야 함. 
            skill_scheduler = self.schedulers['skill_enc_dec']
            msgs = skill_scheduler.step(valid_loss_dict[self.schedulers_metric['skill_enc_dec']])
            self.logger.log(msgs)

            if e >= self.cfg.warmup_steps:
                for module_name, scheduler in self.schedulers.items():
                    if module_name != "skill_enc_dec":
                        msgs = scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])
                        self.logger.log(msgs)

                if self.loop_indicator(e, valid_loss_dict['metric']):
                    print("early stop", 'loss',  valid_loss_dict['metric'])
                    break

            if e > self.cfg.save_ckpt:
                self.save(f'{self.model_id}/{e}.bin')

        self.save(f'{self.model_id}/end.bin')
        self.logger.log('Finished')           
         


    def train_one_epoch(self, loader, e):

        self.meter_initialize()

        start = time.time()

        for i, batch in enumerate(loader):
            self.model.train()

            optim_start = time.time()
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

    def save_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizers' : { module_name : optim['optimizer'].state_dict()  for module_name, optim in self.model.optimizers.items()},
            'schedulers' : { module_name : scheduler.state_dict()  for module_name, scheduler in self.schedulers.items()},
            'configuration' : self.cfg,
            'best_summary_loss': self.best_summary_loss,
            'epoch' : self.e
        }
    

    def save(self, path):
        self.model.eval()
        save_dict = self.save_dict()
        save_dict['cls'] = BaseTrainer
        torch.save(save_dict, path)


    def load(path, cfg = None):
        checkpoint = torch.load(path)
        if cfg is not None:
            self = checkpoint['cls'](cfg)            
            # self = BaseTrainer(cfg)
        else:
            self = checkpoint['cls'](checkpoint['configuration'])            
            # self = BaseTrainer(checkpoint['configuration'])


        self.model.load_state_dict(checkpoint['model'])
        [ optim['optimizer'].load_state_dict(checkpoint['optimizers'][module_name] )  for module_name, optim in self.model.optimizers.items()]
        [ scheduler.load_state_dict(checkpoint['schedulers'][module_name] )  for module_name, scheduler in self.schedulers.items()]
        self.best_summary_loss = checkpoint['best_summary_loss']

        self.e = checkpoint['epoch']

        self.model.eval()

        return self

 
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

class Diversity_Trainer(BaseTrainer):
    def fit(self):
        print("optimizing")
        print(f"log path : {self.run_path}/train.log")
        print(f"weights save path : {self.model_id}")
        rollout  = False

        
        # resume=true and weight가 존재하면 가장 최근 것으로 불러오고, 아래 for loop의 range 조정
        # 없다면 새로 시작한다고 띄우고 ㄱㄱ
        resumed = False

        if self.cfg.resume:
            # resume ckpt weights exists?
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


        # load point가 rollout point보다 클 경우 한번 버퍼 채워야 함. 
        
        if resumed:
            self.logger.log(f"Resumed from {path}")

        if resumed and self.e >= self.cfg.mixin_start - 1:
            # buffer 한번 채워줍시다 
            self.logger.log(f"Filling offline buffer")
            rollout = True 
            self.train_one_epoch(self.train_loader, e, rollout)
            start = deepcopy(self.e) 
        else:
            start = 0




        # for e in range(self.cfg.epochs):
        for e in range(start, self.cfg.epochs):

            start = time.time()
            self.e = e 

            if e >= self.cfg.mixin_start - 1:
                rollout = True


            train_loss_dict  = self.train_one_epoch(self.train_loader, e, rollout)
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


            skill_scheduler = self.schedulers['skill_enc_dec']
            target_metric = self.schedulers_metric["skill_enc_dec"]
            if target_metric is not None:
                msg = skill_scheduler.step(valid_loss_dict[target_metric])
                self.logger.log(msg)

            if "state" in self.schedulers.keys():
                state_scheduler = self.schedulers['state']
                target_metric = self.schedulers_metric["state"]
                if target_metric is not None:
                    msg = state_scheduler.step(valid_loss_dict[target_metric])
                    self.logger.log(msg)

            if e >= self.cfg.warmup_steps:
                for module_name, scheduler in self.schedulers.items():
                    target_metric = self.schedulers_metric[module_name]
                    if module_name not in  ["skill_enc_dec", "state"] and target_metric is not None:
                        msg = scheduler.step(valid_loss_dict[target_metric])
                        self.logger.log(msg)

                if self.loop_indicator(e, valid_loss_dict['metric']):
                    print("early stop", 'loss',  valid_loss_dict['metric'])
                    break

            if e > self.cfg.save_ckpt:
                self.save(f'{self.model_id}/{e}.bin')


            if e == self.cfg.mixin_start:
                self.train_loader.set_mode("with_buffer")

            self.train_loader.update_buffer()
            # seed += 1

        self.save(f'{self.model_id}/end.bin')
        self.logger.log('Finished')           
        


    def train_one_epoch(self, loader, e, rollout):
        # print(loader.dataset.mode)
        self.meter_initialize()
        # start = time.time()
        imgs = None
        for i, batch in enumerate(loader):
            # if i == 0:
            #     print("Loading : ", f"{time.time()-start:.5f}")
            # optim_start = time.time()
            render = False

            if i == 0:
                render = True
        
            self.model.train()
            loss = self.model.optimize(batch, e, rollout, render)

            # if i == 0:
            #     print("Optimize : ", f"{time.time()-optim_start:.5f}")
            if "states_novel" in loss.keys():
                loader.enqueue(loss.pop("states_novel"), loss.pop("actions_novel"), loss.pop("c"))
            
            if "render" in loss.keys():
                imgs = loss.pop("render")

            if not len(self.meters):
                for key in loss.keys():
                    self.meters[key] = AverageMeter()
  
            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])




        if imgs is not None:
            path = f"{self.run_path}/imgs/{e}.mp4"
            print(f"Rendered : {path}")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (400,400))
            for i in range(len(imgs)):
                # writing to a image array
                out.write(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            out.release() 

        return { k : v.avg for k, v in self.meters.items() }

    @torch.no_grad()
    def validate(self, loader, e):

        self.model.eval()
        self.meter_initialize()

        for i, batch in enumerate(loader):
            self.model.eval() # ?
            loss = self.model.validate(batch, e)

            if "states_novel" in loss.keys():
                loss.pop("states_novel")
                loss.pop("actions_novel")

            if not len(self.meters):
                for key in loss.keys():
                    self.meters[key] = AverageMeter()

            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])

        return { k : v.avg for k, v in self.meters.items()}
    
    def save(self, path):
        self.model.eval()
        save_dict = self.save_dict()
        save_dict['cls'] = Diversity_Trainer
        torch.save(save_dict, path)



class Diversity_Trainer2(Diversity_Trainer):
    def train_one_epoch(self, loader, e, rollout):
        # print(loader.dataset.mode)
        self.meter_initialize()
        # start = time.time()
        imgs = None
        self.model.render = False

        for i, batch in enumerate(loader):
            # if i == 0:
            #     print("Loading : ", f"{time.time()-start:.5f}")
            # optim_start = time.time()
            render = False
        
            self.model.train()
            loss = self.model.optimize(batch, e, rollout, render)

            # if i == 0:
            #     print("Optimize : ", f"{time.time()-optim_start:.5f}")
            if "states_novel" in loss.keys():
                loader.enqueue(loss.pop("states_novel"), loss.pop("actions_novel"), loss.pop("c"), loss.pop("seq_indices"))
            
            if "render" in loss.keys():
                imgs = loss.pop("render")

            if not len(self.meters):
                for key in loss.keys():
                    self.meters[key] = AverageMeter()
  
            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])




        if imgs is not None:
            if self.cfg.env_name == "kitchen":
                path = f"{self.run_path}/imgs/{e}.mp4"
                print(f"Rendered : {path}")
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (400,400))
                for i in range(len(imgs)):
                    # writing to a image array
                    out.write(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
                out.release() 
            elif self.cfg.env_name == "maze":
                path = f"{self.run_path}/imgs/{e}.png"
                print(f"Rendered : {path}")
                cv2.imwrite(path, imgs)

        return { k : v.avg for k, v in self.meters.items() }

    @torch.no_grad()
    def validate(self, loader, e):

        self.model.eval()
        self.meter_initialize()

        for i, batch in enumerate(loader):
            self.model.eval() # ?
            loss = self.model.validate(batch, e)

            if "states_novel" in loss.keys():
                loss.pop("states_novel")
                loss.pop("actions_novel")

            if not len(self.meters):
                for key in loss.keys():
                    self.meters[key] = AverageMeter()

            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])

        return { k : v.avg for k, v in self.meters.items()}
    
    def save(self, path):
        self.model.eval()
        save_dict = self.save_dict()
        save_dict['cls'] = Diversity_Trainer2
        torch.save(save_dict, path)
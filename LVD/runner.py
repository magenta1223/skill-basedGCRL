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
        os.makedirs(f"{self.run_path}/data", exist_ok= True)


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
        
        try:
            self.e = checkpoint['epoch']
        except:
            self.e = 0

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

    def eval_rollout(self):
        if self.cfg.env_name == "kitchen":
            self.__eval_rollout_kitchen__()
        elif self.cfg.env_name == "maze":
            self.__eval_rollout_maze__()
        else:
            pass 
    
    @torch.no_grad()
    def __eval_rollout_kitchen__(self):
        os.makedirs(f"{self.run_path}/appendix_figure1/", exist_ok= True)
        
        self.model.eval() # ?
        unseen_goals = self.model.envtask_cfg.unknown_tasks
        seen_goals = set()
        done = False
        

        loader = self.train_loader
        dataset = loader.dataset
        # self.model.render = True
                
        for i, batch in enumerate(self.train_loader):
            # ----------------- Rollout ----------------- #
            batch = edict({  k : v.cuda()  for k, v in batch.items()})
            
            
            result = self.model.rollout(batch)
            # ----------------- Make Trajectory ----------------- #
            states_novel = result['states_novel']
            actions_novel = result['actions_novel']
            seq_indices =  result['seq_indices']
            cs = result['c']

            # unseen G를 달성한 sample을 전부 파악하고
            Gs = [self.model.state_processor.state_goal_checker(state_seq[-1]) for state_seq in states_novel]

            # unseen goals에 포함된 것만 남겨야겠지? 
            # 그것도 유니크한 것만 
            unseen_G_indices = []
            
            for i, G in enumerate(Gs):
                if len(G) == 4 and G in unseen_goals and G not in seen_goals:
                    unseen_G_indices.append(i)
                    seen_goals.add(G)

            # unseen_G_indices = torch.tensor(unseen_G_indices, dtype= torch.bool)
            


            # 각 unseen G에 해당하는 첫 번째 sample을 렌더링 
            # batch 단위로 처리 후 렌더링하는 것으로 변경.. 해야 하지만 concatenated state / action의 길이가 제각각임. 
            for idx in unseen_G_indices:
                state_imgs = []


                states_seq = states_novel[idx]
                actions_seq = actions_novel[idx]
                seq_idx = seq_indices[idx].item()
                c = cs[idx].item()
  
                seq = deepcopy(dataset.seqs[seq_idx])

                # start idx도 필요
                concatenated_states = np.concatenate((seq.states[:c, :dataset.state_dim], states_seq), axis = 0)
                concatenated_actions = np.concatenate((seq.actions[:c], actions_seq), axis = 0)
                concatenated_actions = self.model.action_smoothing(concatenated_states, concatenated_actions)

                G = self.model.state_processor.state_goal_checker(states_seq[-1])
                print(f"Rendering : {G}")

                # State imgs 
                state_imgs = render_from_env(env = self.model.env, task = self.model.tasks[0], states = concatenated_states)
                path = f"{self.run_path}/appendix_figure1/{G}_states.mp4"
                save_video(path,  state_imgs)

                # Action imgs 
                action_imgs = render_from_env(env = self.model.env, task = self.model.tasks[0], states = concatenated_states, actions=  concatenated_actions, c= c)
                path = f"{self.run_path}/appendix_figure1/{G}_actions.mp4"
                save_video(path, action_imgs)

                path = f"{self.run_path}/appendix_figure1/{G}_start.png"
                cv2.imwrite(path, cv2.cvtColor(state_imgs[c], cv2.COLOR_BGR2RGB))

                path = f"{self.run_path}/appendix_figure1/{G}_end_state.png"
                cv2.imwrite(path, cv2.cvtColor(state_imgs[-1], cv2.COLOR_BGR2RGB))

                path = f"{self.run_path}/appendix_figure1/{G}_end_action.png"
                cv2.imwrite(path, cv2.cvtColor(action_imgs[-1], cv2.COLOR_BGR2RGB))

                # 했으면 unseen_goals에서 배제
                unseen_goals.remove(G)

                # 만약 unseen goals가 비었다면 끗 
                if len(unseen_goals) == 0:
                    done = True
                    break
            if done:
                break

        print("Done")

        
    @torch.no_grad()
    def __eval_rollout_maze__(self):

        os.makedirs(f"{self.run_path}/appendix_figure2/", exist_ok= True)        
        self.model.eval() # ?
        loader = self.train_loader
        dataset = loader.dataset

        img = np.rot90(self.model.env.maze_arr != WALL)
        extent = [
            -0.5, self.model.env.maze_arr.shape[0]-0.5,
            -0.5, self.model.env.maze_arr.shape[1]-0.5
        ]

        plt.cla() 
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        # axis options
        ax.axis('off')        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


        target = np.array([12, 20])

        ax.imshow((1-img)/5, extent=extent, cmap='Reds', alpha=0.2)
        ax.scatter(*target, marker='o', c='green', s=30, zorder=3, linewidths=4)
        ax.set_xlim(0, self.model.env.maze_size+1)
        ax.set_ylim(0, self.model.env.maze_size+1)

        


        rollout_candidates = []
        
        for seq in dataset.seqs:
            dists = np.linalg.norm(seq['obs'][:, :2] - target, axis = 1)
            if np.any(dists < 1):
                goal_loc = seq['obs'][-1][:2]
                # ax.scatter(*goal_loc, marker='x', c='red', s=100, zorder=10, linewidths=2)
                states = deepcopy(np.array(seq['obs']))
                ax.plot(*states[:, :2].T  , color='royalblue', linewidth= 3)
                # 해당 seq에서 target위치와 가장 가까운 지점의 index를 고르고
                closest = dists.argmin()
                # 그거만 10개 갖다 붙여서 rollout. 어차피 sequence 쓰지도 않음. 
                rollout_candidates.append(deepcopy(seq['obs'][closest]))

        ax.set_xticks([])
        ax.set_yticks([])

        path = f"{self.run_path}/appendix_figure2/maze_dataset.png"
        fig.savefig(path, bbox_inches="tight", pad_inches = 0)

        # ---------------------------------- # 
        
        # prepare plots 
        plt.cla() 
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        # axis options
        ax.axis('off')        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # set maps 
        ax.imshow((1-img)/5, extent=extent, cmap='Reds', alpha=0.2)
        ax.scatter(*target, marker='o', c='green', s=30, zorder=3, linewidths=4)
        ax.set_xlim(0, self.model.env.maze_size+1)
        ax.set_ylim(0, self.model.env.maze_size+1)

        # states 
        states = torch.tensor(rollout_candidates).cuda()        
        batch = edict(
            states = torch.tensor(rollout_candidates).unsqueeze(1).repeat((1, 10, 1)).cuda()
        )

        result = self.model.prior_policy.rollout(batch)
        states_rollout = result['states_rollout'].detach().cpu().numpy()
        c = result['c']
    
    
        for seq in states_rollout:
            # 싸그리 모아서 batch로 만들고 rollout 
            goal_loc = seq[-1][:2]
            # ax.scatter(*goal_loc, marker='x', c='red', s=100, zorder=10, linewidths=2)
            states = deepcopy(np.array(seq))
            ax.plot(*states[:, :2].T  , color='royalblue', linewidth= 3, alpha= 0.1)

        ax.set_xticks([])
        ax.set_yticks([])
        path = f"{self.run_path}/appendix_figure2/maze_rollout.png"
        fig.savefig(path, bbox_inches="tight", pad_inches = 0)

        print(path)





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
                # cv2.imwrite(path, imgs)
                # plt.Figure 
                imgs.savefig(path, bbox_inches="tight", pad_inches = 0)


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
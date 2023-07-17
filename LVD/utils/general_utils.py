import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from easydict import EasyDict as edict
import datetime


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class ConfigParser:
    def __init__(self) -> None:
        pass

    def __call__(self, v):
        # nested 
        if isinstance(v, DictConfig):
            result = "dict"
        # string, module, not path, not default relative path, and not overrided param 
        elif v == "None" or v == "":
            result = "None"
        # 시부레 처음알았네.. 
        elif self.isClass(v):
            result = "class"
        else:
            result = "value"

        return result
    
    @staticmethod
    def isClass(value):
        # only string can be Class
        if not isinstance(value, str):
            return False
        conds = [
            "." in value, # Is value contains path to Class?
            "/" not in value, # Is value not a path ?
            "=" not in value, # Is value not a overrided hydra hyper parameters?
            value.split(".")[-1][0].isupper() # Does value starts with Capital?
        ]

        return all(conds) 
        

    def __parse_cfg__(self, cfg):
        new_cfg = dict()
        for k, v in cfg.items():
            v_type = self(v)
            if v_type == "dict":
                v = self.__parse_cfg__(v)
            elif v_type == "None":
                v = None
            elif v_type =="class":
                v = hydra.utils.get_class(v)
            else:
                pass
            new_cfg[k] = v
        return new_cfg
    
    def parse_cfg(self, cfg):
        cfg = edict(self.__parse_cfg__(cfg))
        cfg.update(cfg.env)
        cfg.update(cfg.model)
        cfg.update(cfg.rl)

        del cfg.env
        del cfg.model
        del cfg.rl

        return cfg

def get_time():
    now = datetime.datetime.now()
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    return f"{month}{day}-{hour}:{minute}"
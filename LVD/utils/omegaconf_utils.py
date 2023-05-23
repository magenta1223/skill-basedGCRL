from omegaconf import OmegaConf
import numpy as np

def add(*args):
    return sum(args)

def multiply(*args):
    return np.prod(*[args])

def config_path(override_dirname):
    if override_dirname == "":
        override_dirname = "default"
    return override_dirname

def get_trainer(phase, skill_trainer, rl_trainer):
    if phase == "skill":
        return skill_trainer
    else:
        return rl_trainer

OmegaConf.register_new_resolver("add", add)
OmegaConf.register_new_resolver("multiply", multiply)
OmegaConf.register_new_resolver("config_path", config_path)
OmegaConf.register_new_resolver("get_trainer", get_trainer)
from omegaconf import OmegaConf
import numpy as np

def add(*args):
    return sum(args)

def multiply(*args):
    return np.prod(*[args])

def divide(a, b):
    share, remainder = divmod(a,b)

    assert remainder == 0, "배수가 아님"

    return share

def config_path(override_dirname):
    if override_dirname == "":
        override_dirname = "default"
    return override_dirname

def get_trainer(phase, skill_trainer, rl_trainer):
    if phase == "skill":
        return skill_trainer
    else:
        return rl_trainer

def get_cycle(batch_size):
    """
    maintains iters / epoch constantly 
    """
    return batch_size // 20

def get_outdim(latent_state_dim, distributional):
    return latent_state_dim * 2 if distributional else latent_state_dim

def get_indim(manipulation, state_dim, n_obj, skill_dim):
    return n_obj + skill_dim if manipulation else state_dim + skill_dim

def get_statedim(manipulation, state_dim, n_obj):
    return n_obj if manipulation else state_dim 

OmegaConf.register_new_resolver("add", add)
OmegaConf.register_new_resolver("multiply", multiply)
OmegaConf.register_new_resolver("divide", divide)
OmegaConf.register_new_resolver("config_path", config_path)
OmegaConf.register_new_resolver("get_trainer", get_trainer)
OmegaConf.register_new_resolver("get_cycle", get_cycle)
OmegaConf.register_new_resolver("get_outdim", get_outdim)
OmegaConf.register_new_resolver("get_indim", get_indim)
OmegaConf.register_new_resolver("get_statedim", get_statedim)

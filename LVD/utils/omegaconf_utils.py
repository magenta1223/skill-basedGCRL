from omegaconf import OmegaConf
import numpy as np

# def add(a, b):
#   return a + b

# add = lambda x, y : x + y
# multiply = lambda x, y : x * y

def add(*args):
    return sum(args)

def multiply(*args):
    return np.prod(*[args])


def config_path(override_dirname):
    if override_dirname == "":
        override_dirname = "default"
    return override_dirname


# def rl_configs(rl_configs):
#     """
#     전체 config를 받아서 -> rl 용도의 config를 만들기.
#     더 효율적인게 있을텐데? 
#     """
#     #



#     if rl_configs == "":
#         rl_configs = "default"
#     return rl_configs


OmegaConf.register_new_resolver("add", add)
OmegaConf.register_new_resolver("multiply", multiply)
OmegaConf.register_new_resolver("config_path", config_path)
# OmegaConf.register_new_resolver("rl_configs", rl_configs)

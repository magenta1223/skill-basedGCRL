import hydra
from hydra.core.hydra_config import HydraConfig

from LVD.utils import *
from LVD.modules import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict


from LVD.utils.evaluator import Evaluator

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="", version_base= "1.2")
def main(cfg):
    seed_everything(666)
    # print(OmegaConf.to_yaml(cfg))

    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)

    overrides_to_remove = hydra_config.job.override_dirname.split(",") + ['phase=rl', 'eval_mode=learningGraph', 'eval_mode=zeroshot', 'eval_mode=finetuned']
    all_overrides = [override  for override in deepcopy(list(hydra_config.overrides.task))]

    for override in overrides_to_remove:
        if override in all_overrides:
            all_overrides.remove(override)
    rl_overrides = ",".join(all_overrides) # .으로 되어있어서 class로 parser가 class로 인식함.

    with open_dict(cfg):
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.rl_overrides = config_path(rl_overrides)

        if rl_overrides:
            cfg.eval_data_prefix = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/{cfg.rl_overrides}/"
        else:
            cfg.eval_data_prefix = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/default/"


        cfg.eval_rawdata_path = f"{cfg.eval_data_prefix}rawdata.csv"

        # zeroshot
        cfg.zeroshot_weight = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"
        cfg.zeroshot_weight_before_rollout = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/orig_skill.bin"

        # finetune 
        cfg.finetune_weight_prefix = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/sac_{cfg.rl_overrides}"
        cfg.project_name = cfg.structure
        cfg.wandb_run_name = f"{cfg.env.env_name}_{cfg.run_name}_{rl_overrides}"

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    
    evaluator = Evaluator(cfg)
        
    evaluator.evaluate()



if __name__ == "__main__":
    main()

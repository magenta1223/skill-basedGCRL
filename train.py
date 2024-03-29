import hydra
from hydra.core.hydra_config import HydraConfig

from LVD.utils import *
from LVD.modules import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="ours_kitchen", version_base= "1.2")
def main(cfg):
    seed_everything(cfg.seeds[0])
    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)
    # overrides에서 rl_cfgs에 포함된 것만 따로 걸러내야 함. 
    # exclude
    # rl_overrides = "_".join(["".join(override.split(".")[1:]) for override in hydra_config.overrides.task if override != "phase=rl"])
    # rl_overrides = ["".join(override.split(".")[1:]) for override in hydra_config.overrides.task if override != "phase=rl"]

    overrides_to_remove = hydra_config.job.override_dirname.split(",") + ['phase=rl']
    all_overrides = [override  for override in deepcopy(list(hydra_config.overrides.task))]

    for override in overrides_to_remove:
        if override in all_overrides:
            all_overrides.remove(override)
            
    all_overrides = [override for override in all_overrides if "seeds" not in override]

    rl_overrides = ",".join(all_overrides) # .으로 되어있어서 class로 parser가 class로 인식함.


    with open_dict(cfg):
        # COMMON LOGGING PARAMETERS
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.run_id = get_time()
        cfg.rl_overrides = config_path(rl_overrides)

        # PATHS  
        # pretrain weights 
        pretrained_weights = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"
        if not os.path.exists(pretrained_weights):
            pretrained_weights = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/start.bin"
        cfg.skill_weights_path = pretrained_weights

        # rl weights 
        cfg.weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/sac_{cfg.rl_overrides}"
        # rl results 
        cfg.result_path = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/{cfg.rl_overrides}"

        # wandb 
        cfg.project_name = cfg.structure
        cfg.wandb_run_name = f"{cfg.env.env_name}_{cfg.run_name}_{rl_overrides}"




    integrated_cfg_path = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/overrided.yaml"
    OmegaConf.save(cfg, integrated_cfg_path)

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    

    trainer = cfg.trainer_cls(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

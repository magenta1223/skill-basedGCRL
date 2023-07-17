import hydra
from hydra.core.hydra_config import HydraConfig

from LVD.utils import *
from LVD.modules import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="", version_base= "1.2")
def main(cfg):
    seed_everything(cfg.seeds[0])
    # print(OmegaConf.to_yaml(cfg))

    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)
    # overrides에서 rl_cfgs에 포함된 것만 따로 걸러내야 함. 
    # exclude
    # rl_overrides = "_".join(["".join(override.split(".")[1:]) for override in hydra_config.overrides.task if override != "phase=rl"])
    # rl_overrides = ["".join(override.split(".")[1:]) for override in hydra_config.overrides.task if override != "phase=rl"]

    overrides_to_remove = hydra_config.job.override_dirname.split(",") + ['phase=rl']
    all_overrides = [override.replace(".", "_")  for override in deepcopy(list(hydra_config.overrides.task))]
    for override in overrides_to_remove:
        try:
            all_overrides.remove(override)
        except:
            print(f"{override} 없는데요 ? ")
    rl_overrides = ",".join(all_overrides) # .으로 되어있어서 class로 parser가 class로 인식함.

    with open_dict(cfg):
        # COMMON LOGGING PARAMETERS
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.run_id = get_time()
        cfg.rl_overrides = rl_overrides


        # PATHS  
        # pretrain weights 
        cfg.skill_weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"\
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

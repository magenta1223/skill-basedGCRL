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
    seed_everything(cfg.seeds[2])
    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)
    overrides_to_remove = hydra_config.job.override_dirname.split(",") + ['phase=rl']
    all_overrides = [override  for override in deepcopy(list(hydra_config.overrides.task))]

    for override in overrides_to_remove:
        if override in all_overrides:
            all_overrides.remove(override)
    rl_overrides = ",".join(all_overrides) # .으로 되어있어서 class로 parser가 class로 인식함.
    
    with open_dict(cfg):
        # COMMON LOGGING PARAMETERS
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        # PATHS  
        # pretrain weights 
        cfg.skill_weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"\

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    # trainer = cfg.trainer_cls(cfg)
    trainer = cfg.trainer_cls.load(path = cfg.skill_weights_path, cfg = cfg)
    trainer.eval_rollout()

if __name__ == "__main__":
    main()

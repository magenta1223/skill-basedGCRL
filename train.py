import hydra
from hydra.core.hydra_config import HydraConfig

from LVD.utils import *
seed_everything(777)
from LVD.modules import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="", version_base= "1.2")
def main(cfg):
    
    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.skill_weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"
        cfg.weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/sac"
        cfg.project_name = cfg.structure
        cfg.wandb_run_name = f"{cfg.env.env_name}_{cfg.run_name}"


    # integrated_cfg_path = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/overrided.yaml"
    # OmegaConf.save(cfg, integrated_cfg_path)

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    trainer = cfg.trainer_cls(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

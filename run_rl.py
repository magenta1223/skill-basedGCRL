import hydra
from hydra.core.hydra_config import HydraConfig

from LVD.utils import *
seed_everything(777)
from LVD.modules.base import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict
import torch
from LVD.rl.spirl import RL_Trainer

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="spirl_kitchen", version_base= "1.2")
def main(cfg):
    
    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.skill_weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/end.bin"
        cfg.weights_path = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/sac"

        cfg.project_name = cfg.structure
        cfg.wandb_run_name = f"{cfg.env.env_name}_{cfg.run_name}"

    # cfg = OmegaConf.load(cfg_path) # 이게 기존 cfg랑 차이가 있나? 없음. 어차피 똑같은 절차로 만드는거니까. 
    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)

    trainer = RL_Trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()

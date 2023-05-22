import hydra
from hydra.core.hydra_config import HydraConfig


from LVD.utils import *
seed_everything(777)
from LVD.modules.base import *
from LVD.models import *
from LVD.runner import *
from omegaconf import OmegaConf, open_dict

DEFAULT_CONFIGURATION_PATH = "LVD/configs"


@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="spirl_kitchen", version_base= "1.2")
def main(cfg):
    
    hydra_config = HydraConfig.get()
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        
    
    # assert 1==0, OmegaConf.to_yaml(hydra_config)
    # phase 변수를 줘서 하자.
    # phase가 RL이면 -> 이전에 해놨던 뭐시기를 찾아서..
    # 아 싫다. 
    
    # for cfg_source in hydra_config.runtime.config_sources:
    #     if cfg_source.provider == "main":
    #         break

    # if cfg_source.path.replace(hydra_config.runtime.cwd, "") == DEFAULT_CONFIGURATION_PATH:
    #     # skill learning 
    #     integrated_cfg_path = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/overrided.yaml"
    #     OmegaConf.save(cfg, integrated_cfg_path)

    integrated_cfg_path = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/overrided.yaml"
    OmegaConf.save(cfg, integrated_cfg_path)

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)

    trainer = cfg.skill_trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()

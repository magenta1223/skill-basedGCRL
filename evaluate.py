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

    rl_overrides = "_".join(["".join(override.split(".")[1:]) for override in hydra_config.overrides.task if override != "phase=rl"])

    with open_dict(cfg):
        cfg.run_name = config_path(hydra_config.job.override_dirname)
        cfg.job_name = config_path(hydra_config.job.name)
        cfg.rl_overrides = rl_overrides

        cfg.eval_data_prefix = f"logs/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/{cfg.rl_overrides}/"
        cfg.eval_rawdata_path = f"{cfg.eval_data_prefix}/rawdata.csv"

        # zeroshot
        cfg.zeroshot_weight = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/skill/end.bin"
        # finetune 
        cfg.finetune_weight_prefix = f"weights/{cfg.env.env_name}/{cfg.structure}/{cfg.run_name}/sac_{rl_overrides}"
        cfg.project_name = cfg.structure
        cfg.wandb_run_name = f"{cfg.env.env_name}_{cfg.run_name}_{rl_overrides}"

    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    
    evaluator = Evaluator(cfg)
        
    evaluator.evaluate()



if __name__ == "__main__":
    main()

import logging
from rlcarla2.carla_env.collect_data import collect_data
from rlcarla2.carla_env.utils.logger import Logging
import hydra 
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

DEFAULT_CONFIGURATION_PATH = "LVD/configs/data_collecting"

@hydra.main(config_path=DEFAULT_CONFIGURATION_PATH, config_name="carla", version_base= "1.2")
def main(cfg):
    # collect override param 저장
    # logging_path = (config.data_path or Path.cwd()) / "outputs.log"
    
    # logging_path = (cfg.data_path or Path.cwd()) / "outputs.log"

    logging_path = f"{cfg.data_path}/outputs.log"

    print("Logging to", logging_path)
    Logging.setup(
        filepath=logging_path,
        level=logging.DEBUG,
        formatter="(%(asctime)s) [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    collect_data(cfg)

if __name__ == "__main__":
    main()    
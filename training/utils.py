import torch
import hydra
import os 

#### Hydra utils
def get_output_folder_name() -> str:
    """
    Return:
        folder_name: str
            - The name of the folder that holds all the logs/outputs for the current run
            - Not a complete path
    """
    folder_name = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.split("/")[-1]

    return folder_name

def get_output_path() -> str:
    """
    Return:
        output_path: str
            - The absolute path to the folder that holds all the logs/outputs for the current run
    """
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def validate_and_preprocess_cfg(cfg):
    """
    Parameters:
        cfg: DictConfig
            - The hydra config object

    Effects:
        - Sets the logging.run_name to the name of the folder that holds all the logs/outputs for the current run
        - Sets the logging.run_path to the absolute path to the folder that holds all the logs/outputs for the current run
    """
    cfg.logging.run_name = get_output_folder_name()
    cfg.logging.run_path = get_output_path()

    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)
    os.makedirs(os.path.join(cfg.logging.run_path, "checkpoints"), exist_ok=True)


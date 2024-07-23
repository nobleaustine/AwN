# import all requirements
import os
import logging
import torch as th
import torch.distributed as dist
import blobfile as bf
import io
import yaml
from dacite import from_dict
from .script_util import TrainConfig,DiffusionConfig,ModelConfig

def setup_dist_system(batch_size, global_seed):

    assert th.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group(backend="nccl")
    assert batch_size % dist.get_world_size() == 0,"Batch size should be divisible by the number of GPUs."

    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank

    th.manual_seed(seed)
    th.cuda.set_device(device)

    print(f"starting rank:{rank} with seed:{seed}.")
    
    return rank, device

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    
    if dist.get_rank() == 0:  
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def load_config(yaml_file: str='./config/basic_arguments.yaml') -> Config:
       
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    train_config = TrainConfig(**data['train'])
    diffusion_config = DiffusionConfig(**data['diffusion'])
    model_config = ModelConfig(**data['model'])

    return train_config,diffusion_config,model_config

# --- old code ---

def create_argparser():
    # todo: make it pass as an cl argument
    yaml_file_path = './config/basic_arguments.yaml'

    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    defaults = data['train']  
    defaults.update(data['model'])
    defaults.update(data['diffusion'])

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
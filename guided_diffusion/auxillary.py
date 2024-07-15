# import all requirements
import os
import logging
import torch as th
import torch.distributed as dist
import blobfile as bf
import io

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
    # only one process is used to save details
    if dist.get_rank() == 0:  
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    
    return th.load(io.BytesIO(data), **kwargs)
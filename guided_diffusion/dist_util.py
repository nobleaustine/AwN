"""
Helpers for distributed training.
"""

import io
import os
import socket
import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist_system(args):

    assert th.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group(backend="nccl")
    assert args.batch_size % dist.get_world_size()==0,"Batch size should be divisible by the number of GPUs."
    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    th.manual_seed(seed)
    th.cuda.set_device(device) # int(os.environ["LOCAL_RANK"])
    print(f"Starting rank={rank}, device={device}, seed={seed}.")


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda")
#     return th.device("cpu")


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


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p.data,src= 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

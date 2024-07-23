import yaml
import os
from glob import glob
import sys

sys.path.append("../")
sys.path.append("./")
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.auxillary import create_logger,setup_dist_system,load_config

from guided_diffusion.CT_Dataset import NTNUDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from helper_functions.ct_dataloader import CTDataset
from torchvision import transforms

# from guided_diffusion import dist_util, logger
# from visdom import Visdom
# viz = Visdom(port=8850)

def main():
    
    Targs,Dargs,Margs = load_config()
    rank,device = setup_dist_system(Targs.batch_size, Targs.global_seed)

    if dist.get_rank() == 0:
        os.makedirs(Targs.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{Targs.results_dir}/*"))
        model_string_name = Targs.model.replace("/", "-") 
        experiment_dir = f"{Targs.results_dir}/{experiment_index:03d}-{model_string_name}" 
        checkpoint_dir = f"{experiment_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info("setting up dataloader and dataset...")
    transform = transforms.Compose([transforms.Lambda(lambda x: x.to(th.float32))])
    dataset = NTNUDataset(Targs.image_dir, Targs.label_dir, transform)
    process_count = dist.get_world_size()

    sampler = DistributedSampler(
        dataset,
        num_replicas=process_count,
        rank=rank,
        shuffle=True,
        seed=Targs.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(Targs.batch_size //process_count),
        shuffle=False,
        sampler=sampler,
        num_workers=Targs.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({Targs.image_dir})")
    data = iter(loader)

    logger.info("creating model and diffusion...")

    model,diffusion = setup_model_diffusion(Dargs,Margs)
    schedule_sampler = create_named_schedule_sampler(Dargs.schedule_sampler, diffusion,  maxt=Dargs.diffusion_steps)


    logger.info("begin training...")
    TrainLoop(
        rank=rank,
        device=device,
        model=model,
        diffusion=diffusion,
        data=data,
        dataloader=loader,
        batch_size=int(Targs.batch_size //process_count),
        microbatch=Targs.microbatch,
        lr=Targs.lr,
        ema_rate=Targs.ema_rate,
        log_interval=Targs.log_interval,
        save_interval=Targs.save_interval,
        resume_checkpoint=Targs.resume_checkpoint,
        use_fp16=Targs.use_fp16,
        fp16_scale_growth=Targs.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=Targs.weight_decay,
        lr_anneal_steps=Targs.lr_anneal_steps,
    ).run_loop()

if __name__ == "__main__":
    main()

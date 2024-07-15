import yaml
import os
from glob import glob
import sys
import argparse
from omegaconf import DictConfig,OmegaConf
import hydra
sys.path.append("../")
sys.path.append("./")
# from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.auxillary import create_logger,setup_dist_system

from guided_diffusion.CT_Dataset import NTNUDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from helper_functions.ct_dataloader import CTDataset
from torchvision import transforms


def main():

    args = create_argparser()
    rank,device = setup_dist_system(args.batch_size, args.global_seed)

    if dist.get_rank() == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.mod.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}" 
        checkpoint_dir = f"{experiment_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info("setting up dataloader and dataset...")
    transform = transforms.Compose([transforms.Lambda(lambda x: x.to(th.float32))])
    dataset = NTNUDataset(args.image_dir, args.label_dir, transform)
    process_count = dist.get_world_size()

    sampler = DistributedSampler(
        dataset,
        num_replicas=process_count,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size //process_count),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.image_dir})")
    data = iter(loader)

    logger.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.info("begin training...")
    TrainLoop(
        rank=rank,
        device=device,
        model=model,
        diffusion=diffusion,
        data=data,
        dataloader=loader,
        batch_size=int(args.batch_size //process_count),
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


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


if __name__ == "__main__":
    main()

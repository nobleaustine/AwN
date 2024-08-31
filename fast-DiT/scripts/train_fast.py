"""
A minimal training script for DiT.
"""
import torch
import sys
from tqdm import tqdm
import argparse
import yaml

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
sys.path.append('/cluster/home/austinen/NTNU/AwN/SSD/others/fast-DiT/')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from helper_functions.ct_dataloader import NTNUDataset
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models.AWN_fast import awn_models
from diffusion import create_diffusion



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def load_args_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    # Set up distributed training:
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    accelerator = Accelerator()
    device = accelerator.device

    # Set up logger and experiment directory:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}" 
        checkpoint_dir = f"{experiment_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Set up DiT model, EMA, optimizer, and diffusion:
    model = awn_models[args.model](
        input_size=args.image_size
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([transforms.Lambda(lambda x: x.to(torch.float32))])
    dataset = NTNUDataset(args.image_dir, args.label_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True

    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images")


    update_ema(ema, model, decay=0) 
    model.train()  
    ema.eval()  
    model, opt, loader = accelerator.prepare(model, opt, loader)

    train_steps = 0
    log_steps = 0
    running_loss = 0
   
    start_time = time()
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        # y: image, x: label
        for y,x,z in loader:
            x = x.to(device)
            y = y.to(device)       
            # x = x.squeeze(dim=1)
            # y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)
    
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:

                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the YAML config file",default="/cluster/home/austinen/NTNU/AwN/SSD/others/fast-DiT/config/arguments.yaml")

    config_args, remaining_args = parser.parse_known_args()

    if config_args.config:
        yaml_args = load_args_from_yaml(config_args.config)
        parser.set_defaults(**yaml_args)

    # # Add the rest of the arguments
    # parser.add_argument("--data-path", type=str, default="/cluster/home/austinen/NTNU/DATA/training")
    # parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    # parser.add_argument("--image-size", type=int, choices=[64, 256, 512], default=64)
    # parser.add_argument("--epochs", type=int, default=1400)
    # parser.add_argument("--global-batch-size", type=int, default=100)
    # parser.add_argument("--global-seed", type=int, default=0)
    # # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    # parser.add_argument("--num-workers", type=int, default=8)
    # parser.add_argument("--log-every", type=int, default=100)
    # parser.add_argument("--ckpt-every", type=int, default=50_000)

    # Parse all arguments
    args = parser.parse_args(remaining_args)
    main(args)

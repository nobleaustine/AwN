
import torch
import sys
import argparse
import yaml

sys.path.append('/cluster/home/austinen/NTNU/AwN/fast-DiT/')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from helper_functions.CT_dataset import NTNUDataset
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from models.AwN_fast import awn_models
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

def update_args(parser,yaml_file):
    """
    Update the parser with arguments from a yaml file.
    """
    with open(yaml_file, 'r') as file:

        yaml_args = yaml.safe_load(file)

        arguments = {}
        arguments.update(yaml_args["train"])
        arguments.update(yaml_args["model"])
        arguments.update(yaml_args["diffusion"])

        for key, value in arguments.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", type=bool, default=value)
            elif isinstance(value, int):
                parser.add_argument(f"--{key}", type=int, default=value)
            elif isinstance(value, float):
                parser.add_argument(f"--{key}", type=float, default=value)
            elif isinstance(value, str):
                parser.add_argument(f"--{key}", type=str, default=value)
            elif isinstance(value, list):
                parser.add_argument(f"--{key}", type=list, default=value)
            else:
                raise ValueError(f"Unsupported type for argument {key}: {type(value)}")
    

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
    model = awn_models[args.model](input_size=args.image_size)
    model = model.to(device)
    ema = deepcopy(model).to(device)  
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # updated learning rate and weight decay
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
    data_time1 = 0
    d = 0
    f = 0
    b = 0

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    

    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        # y: image, x: label
        data_time1 = time()
        for y, x, z in loader:
            x = x.to(device)
            y = y.to(device)  
            data_time2 = time()
            d = d + data_time2 - data_time1
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            forward_time1 = time()
            with torch.autocast("cuda"):
                loss_dict = diffusion.training_losses_segmentation(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            forward_time2 = time()
            f = f + forward_time2 - forward_time1
            
            
            opt.zero_grad()
            backward_time1 = time()
            # applying gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)
            backward_time2 = time()
            b = b + backward_time2 - backward_time1
        
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                avg_data_time = d / log_steps
                avg_forward_time = f / log_steps
                avg_backward_time = b / log_steps

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    logger.info(f"Data Time: {avg_data_time:.4f}, Forward Time: {avg_forward_time:.4f}, Backward Time: {avg_backward_time:.4f}")
                running_loss = 0
                log_steps = 0
                start_time = time()
                d = 0
                f = 0
                b = 0

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

            data_time1 = time()

    model.eval()
    
    if accelerator.is_main_process:
        logger.info("Done!")

if __name__ == "__main__":

    # parse arguments from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/cluster/home/austinen/NTNU/AwN/fast-DiT/config/arguments.yaml")
    config_args, remaining_args = parser.parse_known_args()

    if config_args.config:
        update_args(parser,config_args.config)
    args = parser.parse_args()
    main(args)

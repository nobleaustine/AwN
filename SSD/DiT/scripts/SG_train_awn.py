
import torch
import sys
sys.path.append('./')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from helper_functions.ct_dataloader import CTDataset,CTImageSegmentationDataset
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

from models.AWN import DiT_models
from diffusion import create_diffusion

#################################################################################
#                             Training Helper Functions                         #
#################################################################################



# fucntion to have a smoothly varying weights
# Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
# doubt: hugely depend on intial weights
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # ema_params = ema_params*decay + (1-decay).model_params
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# intialize gradient calculation
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# destroy all process group in DDP
# def cleanup():
#     """
#     End DDP training.
#     """
#     dist.destroy_process_group()

# info logger
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # only one process is used to save details
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

# image center crop
# def center_crop_arr(pil_image, image_size):
#     """
#     Center cropping implementation from ADM.
#     https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
#     """
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    model = DiT_models[args.model](
        input_size=args.image_size,
    )
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([transforms.Lambda(lambda x: x.to(torch.float32))]) 
    dataset = CTImageSegmentationDataset("/cluster/home/austinen/NTNU/DATA/EDA/IMAGE_SLICES", "/cluster/home/austinen/NTNU/DATA/EDA/LABEL_SLICES",transform=transform)
    
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    d = 0
    tm = 0
    b = 0
    g = 0
    start_time = time()
    d1 = time()
    # --- test --- # addition
    crop_size = 64
    start_index = (512 - crop_size) // 2
    end_index = start_index + crop_size
    # --- test --- # addition
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            # --- test ---
            d2 = time()
            d += d2 - d1
            g1 = time()
            x = x.to(device)
            y = y.to(device)
            g2 = time()
            g = g2 - g1
            x = x[:,:,start_index:end_index, start_index:end_index]
            y = y[:,:,start_index:end_index, start_index:end_index]
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            t1 = time()
            loss_dict = diffusion.training_losses(model,x,y,t)
            loss = loss_dict["loss"].mean()
            t2 = time()
            tm += t2 - t1

            b1 = time()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)
            b2 = time()
            b += b2 - b1

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                data_time = d / log_steps
                train_time = tm / log_steps
                backward_time = b / log_steps
                loading_time = g / log_steps
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() 
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"Data Time: {data_time:.4f}, Train Time: {train_time:.4f}, Backward Time: {backward_time:.4f}, data loading time: {loading_time:.4f}")

                running_loss = 0
                log_steps = 0
                start_time = time()
                d = 0
                tm = 0
                b = 0
                g = 0
                
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            d1 = time()
                

    model.eval()  # important! This disables randomized embedding dropout   
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

if __name__ == "__main__":

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,default="/cluster/home/austinen/NTNU/DATA/training")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=64) # test: from 512
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=8) # test: from 6
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8) # default 4
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
    

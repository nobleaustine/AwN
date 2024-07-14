
# python packages
import sys  
import argparse 
import yaml
import os
from glob import glob 
import logging
# from visdom import Visdom
import torch as th  
from torch.utils import data as DATA
import torchvision.transforms as transforms
import torch.distributed as dist  
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# python module search path: parent directory, current directory 
sys.path.append("/cluster/home/austinen/NTNU/SSD/AWN/")

from helper_functions.resample import create_named_schedule_sampler 
from helper_functions.CT_dataset import NTNUDataset
from helper_functions.train_util import TrainLoop 
from diffusion import create_diffusion
from helper_functions import logger
from helper_functions.script_util import ( 
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)



def main():
    
    args = create_argparser().parse_args() 
    
    assert th.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    th.manual_seed(seed)
    th.cuda.set_device(device)
    print(f"Rank={rank} GPU initialized")

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        os.makedirs(args.out_dir, exist_ok=True)  
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  
        checkpoint_dir = f"{experiment_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.configure(dir=experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    
    logger.info("setting up dataset...")
    train_transform = transforms.Compose([transforms.Lambda(lambda x: x.to(th.float32))])
    dataset = NTNUDataset(args.image_dir,args.label_dir,transform= train_transform) 
    loader = DATA.DataLoader(dataset,batch_size=args.batch_size,shuffle=True) 
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    ) 
    logger.info(f"Dataset contains {len(dataset):,} images ({args.image_dir})") 
    data = iter(loader) 
    logger.info("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(args)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="") 
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log("training...")  

    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=loader,
        batch_size=args.batch_size,
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
        logger=logger,
    ).run_loop()  

def create_argparser():

    yaml_file_path = './MedSegDiff/info/arguments.yaml'

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
    
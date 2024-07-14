
# python packages
import sys  
import argparse 
import yaml
# from visdom import Visdom 

# python module search path: parent directory, current directory 
sys.path.append("/cluster/home/austinen/NTNU/SSD/MedSegDiff/")


from guided_diffusion import dist_util, logger  # ditributed training
from guided_diffusion.resample import create_named_schedule_sampler  # to change the sampling length: 1000

# dataset loaders
# from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
# from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.CT_dataset import NTNUDataset

# scripts utilities
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_model,
)

from guided_diffusion.train_util import TrainLoop # training loop

# pytorch requirements
import torch as th  
import torch.utils.data as t
import torchvision.transforms as transforms
import torch.distributed as dist 

# viz = Visdom(port=8850)  viz : data visulaization

def main():
    #----------------------- setting up distributed training and loading dataset ---------------------------

    args = create_argparser().parse_args()  # args: access arguments from command line
    
    dist_util.setup_dist(args)  #distributed training setup: based on args.multi_gpu
    print("DT: intialized") if dist.is_initialized() else print("DT: NOT intialized")
    
    {
    # logger to save training info to log.txt
    # Configuring the logger to log files into the model output directory
    # logger.configure(dir=args.out_dir)  
    # Logging a message indicating the creation of the data loader
    # logger.log("creating data loader...")  
    
    # creating dataloader based on dataset name
    # if args.data_name == 'ISIC':
        
    #     # transformations
    #     tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    #     transform_train = transforms.Compose(tran_list)

    #     # ds = ISICDataset(args, args.data_dir, transform_train) # dataset
    #     args.in_ch = 4  # input channels count
    # elif args.data_name == 'BRATS':

    #     # transformations
    #     # tran_list = [transforms.Resize((args.image_size,args.image_size)),]
    #     # transform_train = transforms.Compose(tran_list)
        
    #     # ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False) # dataset
    #     # args.in_ch = 5  # input channels count
        
    #     # transformations
    #     tran_list = [transforms.Resize((args.image_size,args.image_size))] #, transforms.ToTensor(),]
    #     transform_train = transforms.Compose(tran_list)
    
    #     ds = CTDataset(args.data_dir, transform_train) # dataset
    #     args.in_ch = 2  # input channels count
    # else :
    #     # Transforming input images using resize and ToTensor transformations
    #     tran_list = [transforms.Resize((args.image_size,args.image_size))]#, transforms.ToTensor(),]
    #     transform_train = transforms.Compose(tran_list)
    #     print("Your current directory : ",args.data_dir)
    #     # Creating CustomDataset object with specified arguments
    #     ds = CTDataset(args, args.data_dir, transform_train)
    #     args.in_ch = 4  # Setting the number of input channels
    }

    print("creating data loader...")

    tran_list = [transforms.Resize((args.image_size,args.image_size))] #, transforms.ToTensor(),]
    transform_train = transforms.Compose(tran_list)

    ds = NTNUDataset(args.image_dir,args.label_dir, transform_train) # dataset
    args.in_ch = 1  # input channels count
     
    dataL= t.DataLoader(ds,batch_size=args.batch_size,shuffle=True)  # datasetloader
      
    data = iter(dataL) # iterable
    #------------------------------------------------------------------------------------------------------

    # logger.log("creating model and diffusion...")  
    print("creating model and diffusion...")

    layer_info = []

    # def print_layer_info(module, input, output):
    #     module_name = module.__name__
    #     input_size = input[0].size() if isinstance(input, tuple) else input.size()
    #     output_size = output.size()
    #     indent = " " * (module.__depth__ * 2)
    #     layer_info.append(f"{indent}{module_name} | Input Size: {input_size} | Output Size: {output_size}")
    #     print(layer_info[-1])
            

    # def register_hooks(module, parent_name="",depth=0):
    #     module_name = parent_name + ('.' if parent_name else '') + (module.__class__.__name__)
    #     module.__name__ = module_name
    #     module.__depth__ = depth    
        
    #     if not list(module.children()):  
    #         module.register_forward_hook(print_layer_info)

    #     named_modules = {m for _, m in module.named_children()}
    #     for n in module.children():
    #         if n not in named_modules:
    #             print("unnamed module detected")
        
    #     for name, child in module.named_children():
    #         child_name = module_name + ('.' if module_name else '') + name
    #         register_hooks(child, child_name,depth + 1)



    # model and diffusion
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))

    if args.multi_gpu:
        # Using multiple GPUs for training if specified
        model = th.nn.DataParallel(model,device_ids=[0]) # int(id) for id in args.multi_gpu.split(',')
        model.to(device = th.device('cuda', int(args.gpu_dev)))
        print("multi processing...")
    else:
        model.to(dist_util.dev())  # Using default device for training

    # exploring: 1
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)

    logger.log("training...")  # Logging a message indicating the start of training

    # def remove_and_join(original_string, substring_to_remove):
    #     start_index = original_string.find(substring_to_remove)
    #     if start_index == -1:
    #         # Substring not found, return original string
    #         return original_string
    #     end_index = start_index + len(substring_to_remove)
    #     # Concatenate the part before the substring and the part after
    #     return original_string[:start_index] + original_string[end_index:]

    # register_hooks(model, model.__class__.__name__,depth=0)
    # input1 = th.randn(1, 3, 512, 512).to(device = th.device('cuda', int(args.gpu_dev)))
    # input2 = th.randn(1,).to(device = th.device('cuda', int(args.gpu_dev)))
    # a = model(input1,input2)
    # with open("layer_info1.txt", "w") as f:
    #     for line in layer_info:
    #         remove_and_join(line,"DataParallel.DataParallel.module.")
    #         remove_and_join(line,"DataParallel.DataParallel.module.")
    #         f.write(line + "\n")  

    # Exploring: 1 Starting the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=dataL,
        batch_size=args.batch_size,
        microbatch=args.microbatch,# memory intensive
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
    ).run_loop()  # Running the training loop


#:defaults+model_and_diffusion_defaults:ArgumentParser(o)
def create_argparser():

    yaml_file_path = './MedSegDiff/info/arguments.yaml'

    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    defaults = data['train'] 
    defaults.update(model_and_diffusion_defaults())  # updating defaults with model and diffusion defaults

    parser = argparse.ArgumentParser()        # creating an argparser to take command line arguments from user
    add_dict_to_argparser(parser, defaults)   # creating arguments for each element in defaults

    return parser  

if __name__ == "__main__":

    main()
    
    # model.to(device="cuda:0")
    # a = model(a)
    # print("hello",a[0].shape)
    # model_graph = draw_graph(model,input_data=a, device='meta')
    # model_graph.visual_graph 
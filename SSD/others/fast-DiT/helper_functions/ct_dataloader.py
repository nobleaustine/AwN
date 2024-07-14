import torch
import torch.nn
import numpy as np
import os
import os.path
import random
from glob import glob
import h5py
import torch.nn.functional as F
from torchvision import transforms

def pad_array(t):
   if t.shape[0] != 640:
    s = (640 - t.shape[0])//2
    padding = ((s,s), (0, 0), (0, 0))
    t = np.pad(t, padding, mode='constant', constant_values=0)
   return t

class CTDataset(torch.utils.data.Dataset):

    def __init__(self, directory, transform, test_flag=False):
 
        super().__init__()
        self.directory = directory # os.path.expanduser(directory)
        self.transform = transform
        self.test_flag=test_flag 
        
        folders = glob(os.path.join(self.directory,"heart_*"))
        folder_paths = [glob(os.path.join(self.directory+"/"+folder.split("/")[-1],"set*")) for folder in folders]

        # all folders
        paths = ["-"+p for folder in folder_paths for p in folder]

        # image 1 and 5
        paths_1 = ["1"+path for path in paths]
        self.data_paths = [item for pair in zip(paths,paths_1) for item in pair]
        random.seed(42)
        random.shuffle(self.data_paths)
        
        # test train split point
        self.ptr= int(0.8*len(self.data_paths))
        if test_flag:
            self.data_paths = self.data_paths[self.ptr:]
        else:
            C_data_paths = ["C"+path for path in self.data_paths[:self.ptr]]
            self.data_paths = [item for pair in zip(self.data_paths[:self.ptr], C_data_paths) for item in pair]
       
        
    def __len__(self):
        return len(self.data_paths)*640
    

    def __getitem__(self, x):
        try:
            n = x // 640
            slice = x % 640
            path   = self.data_paths[n]
            check,pathed = path.split("-")
            with h5py.File(pathed, 'r') as f:
                if check == "C1":
                    img = f["raw"][0]
                    lab = f["label"][0]
                elif check == "1":
                    img = f["raw"][1]
                    lab = f["label"][0]
                elif check == "C":
                    img = f["raw"][2]
                    lab = f["label"][1]
                else:
                    img = f["raw"][3]
                    lab = f["label"][1]
            image = pad_array(img)
            label = pad_array(lab)
            image = image[slice,...]
            label = label[slice,...]
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            return label,image
        except Exception as e:
            print(f"Error in worker {os.getpid()}: {e}") 
            raise                                                                                                                                                                                                                                                                                                             

if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor()])
    d = CTDataset("/cluster/home/austinen/NTNU/DATA/training",transform)
    print(len(d))
    # y,x = d[0]
    # y,x = d[700]
    # y,x = d[1500]
    # y,x = d[2000]

  

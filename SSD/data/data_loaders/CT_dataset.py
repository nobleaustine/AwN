# import nibabel as nib
# import pydicom

# import scipy
# import glob
# from collections import Counter
# import matplotlib.pyplot as plt
# import numpy as np
# import os
import h5py
from torch.utils.data import Dataset
from torchvision import transforms 
from transforms3d import *

class CTDataset(Dataset):
    
    def __init__(self,img_dir, aug=None, mode='train',
                     channel_size_3d=1, ct_slice_dim=512,organ="heart"):

        Dataset.__init__(self)

        self.img_dir = img_dir
        self.data_paths = []
        self.mode = mode
        self.channel_size_3d = channel_size_3d
        self.ct_slice_dim = ct_slice_dim

        for i in range(1,26,5):
            for j in range(5):
                path = img_dir + f'/{organ}_{i}_{i+4}/set_{i+j}.h5'
                self.data_paths.append(path + '1c')
                self.data_paths.append(path + '1n')
                self.data_paths.append(path + '5c')
                self.data_paths.append(path + '5n')
                 
        spl = [.8,.1,.1]
        
        train_ptr = int(spl[0]*len(self.data_paths)) 
        val_ptr = train_ptr + int(spl[1]*len(self.data_paths))
 
        if aug:
            train_transforms = transforms.Compose([
                                ToPILImage3D(),
                                Resize3D((ct_slice_dim, ct_slice_dim)),
                                transforms.RandomChoice([
                                    RandomHorizontalFlip3D(),
                                    RandomVerticalFlip3D(),
                                    RandomRotation3D(30),
                                    RandomShear3D(45, translate=.4, scale=(.7,1.3), shear=.2)]), 
                                ToTensor3D(),
                                Normalize3D('min_max')])

            train_unnormalized = transforms.Compose([
                                ToPILImage3D(),
                                Resize3D((ct_slice_dim, ct_slice_dim)),
                                transforms.RandomChoice([
                                    RandomHorizontalFlip3D(),
                                    RandomVerticalFlip3D(),
                                    RandomRotation3D(30)]),
                                ToTensor3D(),])
        else:
            train_transforms = transforms.Compose([
                                ToPILImage3D(),
                                Resize3D((ct_slice_dim, ct_slice_dim)),
                                ToTensor3D(),
                                Normalize3D('min_max')])

            train_unnormalized = transforms.Compose([
                                ToPILImage3D(),
                                Resize3D((ct_slice_dim, ct_slice_dim)),
                                ToTensor3D(),])

        val_transforms = transforms.Compose([
                                ToPILImage3D(),
                                Resize3D((ct_slice_dim, ct_slice_dim)),
                                ToTensor3D(),
                                IndividualNormalize3D(),])

        if mode == 'train':
            self.data_paths = self.data_paths[:train_ptr]
            self.transforms = train_transforms
            self.seg_transforms = train_unnormalized
        elif mode == 'val':
            self.data_paths = self.data_paths[train_ptr:val_ptr]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized
        else:
            self.data_paths = self.data_paths[val_ptr:]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx][:-2]
        num = self.data_paths[idx][-2:]

        with h5py.File(path, 'r') as f:
            
            if num == "1c":
                raw = f["raw"][0]
                label = f["label"][0]
            elif num == "1n":
                raw = f["raw"][1]
                label = f["label"][0]
            elif num == "5c":
                raw = f["raw"][2]
                label = f["label"][1]
            elif num == "5n":
                raw = f["raw"][3]
                label = f["label"][1]

        inp, out = torch.tensor(raw), torch.tensor(label)     
        
        inp = inp.permute(2, 0, 1)
        out = out.permute(2, 0, 1)
        
        if inp.size(0)<=self.channel_size_3d:
            batch_size = (self.channel_size_3d,) + inp.size()[1:]
            temp1, temp2 = torch.zeros(batch_size), torch.zeros(batch_size)
            temp1[:inp.size(0),:,:] = inp
            temp2[:out.size(0),:,:] = out

            inp, out = temp1, temp2
            
        else:
            r = np.random.randint(0, inp.size(0)-self.channel_size_3d)
            inp, out = inp[r:r+self.channel_size_3d,:,:], out[r:r+self.channel_size_3d,:,:]
        
        if self.transforms:
            inp, out = self.transforms(inp), self.seg_transforms(out) 
        
        out[out>0] = 1
        
        return inp, out

    
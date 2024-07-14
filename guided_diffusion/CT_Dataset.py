import os
# from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class NTNUDataset(Dataset):
    
    def __init__(self, image_dir, label_dir, transform=None,mode= 'Training'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data_list = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
        self.mode = mode
        # images = sorted(glob(os.path.join(image_dir, '.nii.gz')))
        # labels = sorted(glob(os.path.join(label_dir, '.nii.gz')))
        # print(f"Found {len(self.data_list)} images in {image_dir}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):

        try:
            name = self.data_list[idx]
            image_path = os.path.join(self.image_dir,name)
            label_path = os.path.join(self.label_dir,name)

            image_obj = nib.load(image_path)
            label_obj = nib.load(label_path)
            
            image = torch.from_numpy(np.asarray(image_obj.dataobj))
            label = torch.from_numpy(np.asarray(label_obj.dataobj))
            
            # adding channel dimension
            image_slice = image.unsqueeze(0).float()

            # one-hot encoding
            label_slice = torch.zeros((3, label.shape[0], label.shape[1]))  
            for i in range(3): 
                label_slice[i] = (label == i).float()
            
            
            if self.transform:
                image_slice = self.transform(image_slice)
                label_slice = self.transform(label_slice)

            if self.mode == 'Training':
                return (image_slice, label_slice, name)
            else:
                return (image_slice, label_slice, name)
                
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None



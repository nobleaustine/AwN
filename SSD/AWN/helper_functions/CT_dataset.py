import torch
import numpy as np
import os
import nibabel as nib
from torch.utils.data import Dataset

class NTNUDataset(Dataset):
    
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
        # print(f"Found {len(self.image_filenames)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):

        try:
            image_path = os.path.join(self.image_dir, self.image_filenames[idx])
            label_path = os.path.join(self.label_dir, self.image_filenames[idx])

            image_obj = nib.load(image_path)
            label_obj = nib.load(label_path)
            
            image = torch.from_numpy(np.asarray(image_obj.dataobj))
            label = torch.from_numpy(np.asarray(label_obj.dataobj))
            
            # adding channel dimension
            image_slice = image.unsqueeze(0).float()
            label_slice = label.unsqueeze(0).float()  
            
            if self.transform:
                image_slice = self.transform(image_slice)
                label_slice = self.transform(label_slice)
            
            return label_slice,image_slice
        
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None
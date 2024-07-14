
import torch
import numpy as np
import os
import nibabel as nib
from torch.utils.data import Dataset

{
# import torchvision.utils as vutils
# from glob import glob
# import h5py
# import torch.nn.functional as F

# def pad_array(t):
#    s = (640 - t.shape[0])//2
#    padding = ((s,s), (0, 0), (0, 0))
#    t = np.pad(t, padding, mode='constant', constant_values=0)
#    return t

# class CTDataset(torch.utils.data.Dataset):

#     def __init__(self, directory, transform, test_flag=False):
 
#         super().__init__()
#         self.directory = os.path.expanduser(directory)
#         self.transform = transform
#         self.test_flag=test_flag   # for testing segmentation not included
        
#         # data channels or types
#         if test_flag:
#             self.seqtypes = ['w']
#         else:
#             self.seqtypes = ['c','w','s']

#         self.seqtypes_set = set(self.seqtypes)
        
#         folders = glob(os.path.join(self.directory,"heart_*"))
#         folder_paths = [glob(os.path.join(self.directory+"/"+folder.split("/")[-1],"set*")) for folder in folders]
#         paths = [p for folder in folder_paths for p in folder]
#         paths_5 = ["5B"+path for path in paths]
#         self.database = paths + paths_5
       
        
#     def __len__(self):
#         return len(self.database)*640
    
#     def __getitem__(self, x):
#         out = []
#         n = x // 640
#         slice = x % 640
#         path   = self.database[n]
#         for seqtype in self.seqtypes:
#             if path.split("B")[0] == "5":
#                 with h5py.File(path.split("B")[1], 'r') as f:
#                     if seqtype == "c":
#                         img= f["raw"][2]
#                     elif seqtype == "w":
#                         img= f["raw"][3]
#                     else:
#                         img = f["label"][1]
#             else:
#                 with h5py.File(path, 'r') as f:
#                     if seqtype == "c":
#                         img= f["raw"][0]
#                     elif seqtype == "w":
#                         img= f["raw"][1]
#                     else:
#                         img = f["label"][0]
#             img = pad_array(img)
#             o = torch.tensor(img)[slice,:,:]
#             out.append(o)
#         out = torch.stack(out)
        
#         if self.test_flag:
#             image=out
#             if self.transform:
#                 image = self.transform(image)
#             return (image, image, path + "_slice" + str(slice))
#         else:

#             image = out[:-1, ...]
#             label = out[-1, ...][None, ...]
            
#             label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
#             if self.transform:
#                 state = torch.get_rng_state()
#                 image = self.transform(image)
#                 torch.set_rng_state(state)
#                 label = self.transform(label)
#             return (image, label, path + "_slice" + str(slice)) # virtual path
}

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
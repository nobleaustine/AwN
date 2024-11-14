import sys
sys.path.append('/cluster/home/austinen/NTNU/AwN/fast-DiT/')
from time import time

from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions.CT_dataset import NTNUDataset
import torch


transform = transforms.Compose([transforms.Lambda(lambda x: x.to(torch.float32))])
dataset = NTNUDataset("/cluster/home/austinen/NTNU/DATA/EDA/IMAGE_SLICES/", "/cluster/home/austinen/NTNU/DATA/EDA/LABEL_SLICES/", transform)
loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True

    )
d = 0
c = 100
s = time()
for x,y,z in loader:
    a = x.squeeze().numpy()
    b = y.squeeze().numpy()

    e = time()
    d = d + e-s 
    c = c-1
    s = time()
    if c == 0:
        print("average time: ", d/100)
        break
   

# view images and label
# plt.figure(figsize=(10,5))
# plt.subplot(1,3,1)
# plt.imshow(a, cmap='gray')
# plt.title('Image')

# plt.subplot(1,3,2)
# plt.imshow(b1, cmap='gray')
# plt.title('Label1')
# plt.subplot(1,3,3)
# plt.imshow(b2, cmap='gray')
# plt.title('Label2')

# plt.show()
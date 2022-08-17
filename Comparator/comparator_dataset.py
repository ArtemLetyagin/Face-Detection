from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import numpy as np

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transforms=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transforms
    
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        
        same_class = random.randint(0,1)
        if same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
            
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
    
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return 1000
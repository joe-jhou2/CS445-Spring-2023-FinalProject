

#code taken from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/CycleGAN/dataset.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class GrumpyCatDataset(Dataset):
    def __init__(self, A_dir, B_dir):
        
        self.dataA = A_dir
        self.dataB = B_dir
        self.transform = transforms = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )
        
        self.A_images = os.listdir(A_dir)
        self.B_images = os.listdir(B_dir)
        self.A_len = len(self.dataA)
        self.B_len = len(self.dataB)

    def __len__(self):
        return max(len(self.dataA), len(self.dataB)) 

    def __getitem__(self, index):
      
        A_path = os.path.join(self.dataA, self.A_images[index % self.A_len])
        B_path = os.path.join(self.dataB, self.B_images[index % self.B_len])

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=A_img, image0=B_img)
            A_img = augmentations["image"]
            B_img = augmentations["image0"]

        return A_img, B_img

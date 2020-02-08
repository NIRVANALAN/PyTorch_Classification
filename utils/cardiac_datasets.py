from __future__ import print_function, division
import os
import pdb
import io

import torch
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return np.array(img)

class CardiacDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datalist, root_dir='', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.datalist = open(datalist).readlines()
        

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.datalist[idx].rstrip().split()
        label = int(label)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        # sample = {'image': image, 'landmarks': landmarks}

        return image, label




import shutil

import time

import pandas as pd
import numpy as np

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import normalize
from torch.utils.data import random_split
from torchvision.transforms import v2
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.utils import HAM10000Segmentation, transform_seg


def get_segmentation_dataloader(image_dir: str, mask_dir: str) -> tuple:
    dataset_seg = HAM10000Segmentation(image_dir, mask_dir, transform=transform_seg)
    train_dataset_seg, test_dataset_seg = torch.utils.data.random_split(dataset_seg, [0.8, 0.2])
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=64, shuffle=True)
    test_loader_seg = torch.utils.data.DataLoader(test_dataset_seg, batch_size=64, shuffle=True)

    return train_loader_seg, test_loader_seg
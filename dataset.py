import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
# Create a custom dataset class
# TEMPLATE, TODO: Implement
class AudioDataset(Dataset):
    def __init__(self, audio_dir, targets, transform=None, target_transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        sample = None
        target = None
        filename = None

        return sample, target, filename
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio
# Create a custom dataset class

class AudioDataset(Dataset):
    def __init__(self, content_file, audio_dir):
        #Data loading
        self.content = pd.read_csv(content_file, sep='\t')
        self.audio_dir = audio_dir
    def __len__(self):
        #Length of the dataset
        return len(self.content)
    
    def __getitem__(self, index):
        #Get item: returns the sample and its corresponding label
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    def _get_audio_sample_path(self, index):
        #Get audio sample path
        #Returns the path of the audio sample in the index
    
        path = os.path.join(self.audio_dir, self.content.iloc[index, 0].replace('audio/', ''))
        return path

    def _get_audio_sample_label(self, index):
        #Get audio sample label
        #Returns the label of the audio sample in the index
        return self.content.iloc[index, 1]
    
    
if __name__ == "__main__":
    # Test the dataset
    dataset = AudioDataset('data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', 'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio')
    print(f"Hay un total de {len(dataset)} muestras en el dataset")
    signal, label = dataset[0]
    #Print signal and label
    print(f"La señal es de tipo {signal} y label {label}")


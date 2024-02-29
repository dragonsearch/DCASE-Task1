import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
#import dataset.base_dataset as base_dataset
from dataset.base_dataset import Base_dataset
class Cached_dataset(Base_dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device, label_encoder, cache_transforms=True):
        """
        The function initializes an object with a content file and an audio directory.
        
        :param content_file: The content_file parameter is the file path to a CSV file that contains the
        content data. The data in the CSV file is separated by tabs ('\t')
        :param audio_dir: The audio directory where the audio files are stored
        :param transformations: The transformations parameter is a transformation object that is used to
        transform the audio samples
        :param sample_rate_target: The sample_rate_target parameter is the target sample rate of the
        audio samples
        :param device: The device parameter is the device where the audio samples will be stored. It can
        be either 'cpu' or 'cuda'
        :param cache_transforms: The cache_transforms parameter determines whether the transformations
        are already cached to disk. If the transformations are already cached, then this parameter should be False
        """
       
        super().__init__(content_file, audio_dir, transformations , sample_rate_target, device, label_encoder)
        if not os.path.exists('data/cache'):
            os.makedirs('data/cache')
        if cache_transforms:
            self._cache_transforms()
        #self._test_cached_transforms()

    def _transform_data(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        filename = self._get_audio_sample_filename(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)
        signal = self.transformations(signal)
        return signal, sr, filename

    def _test_cached_transforms(self):
        """
        The function `_test_cached_transforms` is used to test the cached transformations to make sure
        that they were saved correctly.
        
        :param signal: The `signal` parameter is the audio signal that needs to be transformed
        :param sr: The `sr` parameter is the sample rate of the audio signal
        :return: None
        """
        for i in range(0, len(self.content)):
            audio_sample_path = self._get_audio_sample_path(i)
            filename = self._get_audio_sample_filename(i)
            signal, sr = torchaudio.load(audio_sample_path)
            signal = signal.to(self.device)
            #resample and mixdown if necessary (assuming dissonance in the dataset)
            signal = self._resample_if_needed(signal, sr)
            signal = self._mix_down_if_needed(signal)
            signal = self.transformations(signal)
            cached_signal = torch.load(f'data/cache/{filename}.pt')
            assert torch.equal(signal, cached_signal)
        
        print("All cached signals are equal to the transformed signals")

    def _cache_transforms(self):
        """
        The function `_cache_transforms` is used to cache the transformations
        to disk so that they can be reused later.
        
        :param signal: The `signal` parameter is the audio signal that needs to be transformed
        :param sr: The `sr` parameter is the sample rate of the audio signal
        :return: None
        """
        for i in range(0, len(self.content)):
            signal, sr, filename = self._transform_data(i)
            torch.save(signal, f'data/cache/{filename}.pt')
        print("Transformations are cached to disk")
    
    def __getitem__(self, index):
        """
        The `__getitem__` function returns the audio signal and label for a given index in a dataset.
        
        :param index: The `index` parameter represents the index of the audio sample that you want to
        retrieve from your dataset
        :return: The `__getitem__` method is returning a tuple containing the audio signal and the label
        of the audio sample at the specified index.
        """
        filename = self._get_audio_sample_filename(index)
        label = self._get_audio_sample_label(index)
        signal = torch.load(f'data/cache/{filename}.pt')
        
        return signal, label, filename
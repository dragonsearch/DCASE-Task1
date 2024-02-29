import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


class Eval_dataset(Dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device):
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
        """
       
        self.content = pd.read_csv(content_file, sep='\t')
        self.audio_dir = audio_dir
        self.device = device
        self.transformations = transformations.to(self.device)
        self.sample_rate_target = sample_rate_target
    def __len__(self):
        """
        The above function returns the length of the content attribute of an object.
        :return: The length of the content attribute of the object.
        """
        
        return len(self.content)
    def __getitem__(self, index):
        """
        The `__getitem__` function returns the audio signal and label for a given index in a dataset.
        
        :param index: The `index` parameter represents the index of the audio sample that you want to
        retrieve from your dataset
        :return: The `__getitem__` method is returning a tuple containing the audio signal and the label
        of the audio sample at the specified index.
        """
        
        audio_sample_path = self._get_audio_sample_path(index)
        filename = self._get_audio_sample_filename(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)
        signal = self.transformations(signal)

        return signal, filename
    
    
    def _get_audio_sample_path(self, index):
        """
        The function `_get_audio_sample_path` returns the path of an audio sample based on its index and
        the audio directory.
        
        :param index: The index parameter is the index of the audio sample in the dataset. It is used to
        locate the audio sample in the dataset and retrieve its path
        :return: the path of an audio sample file.
        """
       
    
        path = os.path.join(self.audio_dir, self.content.iloc[index, 0].replace('audio/', ''))
        return path

    def _get_audio_sample_filename(self, index):
        """
        The function `_get_audio_sample_filename` returns the filename of an audio sample at a given
        index.
        
        :param index: The index parameter is the index of the row in the content dataframe that you want
        to retrieve the audio sample filename from
        :return: the value in the third column (index 2) of the DataFrame "content" at the specified
        index.
        """
    
        return self.content.iloc[index, 0].replace('audio/', '')


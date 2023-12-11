import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio



# The `AudioDataset` class is a custom dataset class for loading audio samples and their corresponding
# labels from a content file and audio directory.

class AudioDataset(Dataset):
    def __init__(self, content_file, audio_dir, transformations ,sample_rate_target):
        """
        The function initializes an object with a content file and an audio directory.
        
        :param content_file: The content_file parameter is the file path to a CSV file that contains the
        content data. The data in the CSV file is separated by tabs ('\t')
        :param audio_dir: The audio directory where the audio files are stored
        """
       
        self.content = pd.read_csv(content_file, sep='\t')
        self.audio_dir = audio_dir
        self.transformations = transformations
        self.sample_rate_target = sample_rate_target

        
    def __len__(self):
        """
        The above function returns the length of the content attribute of an object.
        :return: The length of the content attribute of the object.
        """
        
        return len(self.content)
    
    def _resample_if_needed(self, signal, sr):
        """
        The function resamples an audio signal if its sample rate is different from the target sample
        rate.
        
        :param signal: The `signal` parameter is the audio signal that needs to be resampled. It is
        typically represented as a 1-dimensional tensor or array
        :param sr: The parameter `sr` represents the sample rate of the input signal
        :return: the resampled signal.
        """
        if sr != self.sample_rate_target:

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate_target)
            signal = resampler(signal)
      
        return signal
    
    def _mix_down_if_needed(self, signal):
        """
        The function `_mix_down_if_needed_` takes a multi-channel audio signal and returns a
        single-channel signal by taking the mean across channels.
        
        :param signal: The signal parameter is a tensor representing an audio signal. It has a shape of
        (num_channels, num_samples), where num_channels is the number of audio channels (e.g., 1 for
        mono, 2 for stereo) and num_samples is the number of audio samples
        :param sr: The parameter "sr" stands for sample rate, which is the number of samples per second
        in the audio signal
        :return: the signal after applying the mean operation along the first dimension (dim=0) and
        keeping the dimension (keepdim=True).
        """

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

   
        return signal
    
    def __getitem__(self, index):
        """
        The `__getitem__` function returns the audio signal and label for a given index in a dataset.
        
        :param index: The `index` parameter represents the index of the audio sample that you want to
        retrieve from your dataset
        :return: The `__getitem__` method is returning a tuple containing the audio signal and the label
        of the audio sample at the specified index.
        """
        
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)
        signal = self.transformations(signal)

        return signal, label
    
    
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

    def _get_audio_sample_label(self, index):
        """
        The function `_get_audio_sample_label` returns the label of an audio sample at a given index.
        
        :param index: The index parameter is the index of the row in the content dataframe that you want
        to retrieve the audio sample label from
        :return: the value in the second column (index 1) of the DataFrame "content" at the specified
        index.
        """
    
        return self.content.iloc[index, 1]
    
    
if __name__ == "__main__":

    SAMPLE_RATE = 16000
    # Test the dataset

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = 48000,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    )

    dataset = AudioDataset('data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', 'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio', mel_spectrogram, SAMPLE_RATE)

    print(f"Hay un total de {len(dataset)} muestras en el dataset")
    signal, label = dataset[0]
    print(f"La primera muestra tiene un shape de {signal.shape} y su label es {label}")
    print (signal)
    


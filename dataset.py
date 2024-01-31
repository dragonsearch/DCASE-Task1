import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import warnings

# The `AudioDataset` class is a custom dataset class for loading audio samples and their corresponding
# labels from a content file and audio directory.

class AudioDataset(Dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device, label_encoder):
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
        self.label_encoder = label_encoder
        self.encoded_labels = self.content.copy()
        self.encoded_labels.iloc[:, 1] = self.label_encoder.fit_transform(self.content.iloc[:, 1])
        
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

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate_target).to(self.device)
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
        filename = self._get_audio_sample_filename(index)
        label = self._get_audio_sample_label(index)
        identifier = self._get_audio_sample_identifier(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)

        signal = self.transformations(signal)
        
        return signal, label, filename, identifier
     
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
    
        return self.encoded_labels.iloc[index, 1]
    
    def _get_audio_sample_identifier(self, index):
        """
        The function `_get_audio_sample_identifier` returns the identifier of an audio sample at a given
        index.
        
        :param index: The index parameter is the index of the row in the content dataframe that you want
        to retrieve the audio sample identifier from
        :return: the value in the first column (index 0) of the DataFrame "content" at the specified
        index.
        """
    
        return self.content.iloc[index, 2]
    
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

class AudioDataset_with_tensorboard(Dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device, label_encoder):
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
        self.label_encoder = label_encoder
        self.encoded_labels = self.content.copy()
        self.encoded_labels.iloc[:, 1] = self.label_encoder.fit_transform(self.content.iloc[:, 1])

        #Tensorboard writer

        self.writer = SummaryWriter(log_dir='DatasetAudio')
        self.audio_samples_added = {}
        
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

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate_target).to(self.device)
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
    
    def _save_audio_to_tensorboard(self, audio, label, filename):
        print(f'Addinnng audio sample {filename} to TensorBoard')
        if label not in self.audio_samples_added:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(f'Adding audio sample {filename} to TensorBoard')
                self.writer.add_audio(f'Audio/{label}/{filename}', audio, sample_rate=self.sample_rate_target)
            self.audio_samples_added[label] = filename

    def _save_mel_spectrogram_to_tensorboard(self, mel_spectrogram, label, filename):
        if label in self.audio_samples_added and self.audio_samples_added[label] == filename:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.imshow(torch.log(mel_spectrogram[0].cpu() + 1e-9).numpy(), cmap='viridis', aspect='auto', origin='lower')
                plt.title(f'Mel Spectrogram - {label} - {filename}')
                plt.xlabel('Time')
                plt.ylabel('Mel Filter')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
            print(f'Adding mel spectrogram {filename} to TensorBoard')
            self.writer.add_figure(f'Mel_Spectrogram/{label}/{filename}', plt.gcf())
    
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
        label = self._get_audio_sample_label(index)
        identifier = self._get_audio_sample_identifier(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)

        # Save audio and mel spectrogram to TensorBoard
        self._save_audio_to_tensorboard(signal, label, filename)

        signal = self.transformations(signal)

        self._save_mel_spectrogram_to_tensorboard(signal, label, filename)
        return signal, label, filename, identifier
    
    
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
    
        return self.encoded_labels.iloc[index, 1]
    
    def _get_audio_sample_identifier(self, index):
        """
        The function `_get_audio_sample_identifier` returns the identifier of an audio sample at a given
        index.
        
        :param index: The index parameter is the index of the row in the content dataframe that you want
        to retrieve the audio sample identifier from
        :return: the value in the first column (index 0) of the DataFrame "content" at the specified
        index.
        """
    
        return self.content.iloc[index, 2]
    
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

class AudioDataset_fold(Dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device, label_encoder):
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
        self.label_encoder = label_encoder
        self.encoded_labels = self.content.copy()
        self.encoded_labels.iloc[:, 1] = self.label_encoder.fit_transform(self.content.iloc[:, 1])
        
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

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate_target).to(self.device)
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
        filename = self._get_audio_sample_filename(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)

        signal = self.transformations(signal)
        
        return signal, label, filename
     
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
    
        return self.encoded_labels.iloc[index, 1]
    
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

class AudioDatasetEval(Dataset):
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

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate_target).to(self.device)
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
        filename = self._get_audio_sample_filename(index)
        label = self._get_audio_sample_label(index)
        identifier = self._get_audio_sample_identifier(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)
        signal = self.transformations(signal)

        return signal, label, filename, identifier
    
    
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
    
        return self.content.iloc[index, 0]
    
    def _get_audio_sample_identifier(self, index):
        """
        The function `_get_audio_sample_identifier` returns the identifier of an audio sample at a given
        index.
        
        :param index: The index parameter is the index of the row in the content dataframe that you want
        to retrieve the audio sample identifier from
        :return: the value in the first column (index 0) of the DataFrame "content" at the specified
        index.
        """
    
        return self.content.iloc[index, 0]
    
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

if __name__ == "__main__":

    SAMPLE_RATE = 16000
    # Test the dataset

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    )
    
    dataset = AudioDataset_with_tensorboard('data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', 'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio', mel_spectrogram, 22050, 'cuda', label_encoder=LabelEncoder())
    print(f'Tensorboard logs are saved in {dataset.writer.log_dir}')
    print(f'Number of samples in the dataset: {len(dataset)}')
    print(f'Number of classes: {len(dataset.label_encoder.classes_)}')
    print(f'Classes: {dataset.label_encoder.classes_}')
    print(f'Sample rate: {dataset.sample_rate_target}')
    """
    dataset_fold = AudioDataset_fold('data/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_train.csv', 'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio', mel_spectrogram, 22050, 'cuda', label_encoder=LabelEncoder())
    print(f'Number of samples in the dataset: {len(dataset_fold)}')
    print(f'Number of classes: {len(dataset_fold.label_encoder.classes_)}')
    print(f'Classes: {dataset_fold.label_encoder.classes_}')
    print(f'Sample rate: {dataset_fold.sample_rate_target}')
    """

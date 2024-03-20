import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import numpy as np 
import pandas as pd
import os
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift
#import dataset.base_dataset as base_dataset
from dataset.base_dataset import Base_dataset

class Shifted_dataset(Base_dataset):
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
        El método __getitem__ devuelve las 10 muestras de 1 segundo correspondientes a un segmento de 10 segundos
        del audio original, aplicando un desplazamiento temporal al conjunto de muestras.
        
        :param index: El índice de la muestra de 10 segundos que se va a recuperar.
        :return: Un tensor que contiene las 10 muestras de 1 segundo correspondientes al segmento de 10 segundos
                del audio original, con un desplazamiento temporal aplicado.
        """
        
        # Obtener la ruta del archivo de audio y el nombre del archivo
        audio_sample_path = self._get_audio_sample_path(index)
        filename = self._get_audio_sample_filename(index)
        
        # Cargar las 10 muestras de 1 segundo correspondientes al segmento de 10 segundos del audio original
        segment_signal = []
        segment_label = []
        for i in range(10):
            signal, sr = torchaudio.load(audio_sample_path)  # Cargar una muestra de 1 segundo
            label = self._get_audio_sample_label(index)  # Obtener la etiqueta de la muestra
            signal = signal.to(self.device)
            # Resample y mixdown si es necesario
            signal = self._resample_if_needed(signal, sr)
            signal = self._mix_down_if_needed(signal)
            signal = self.transformations(signal)  # Aplicar transformaciones
            
            segment_signal.append(signal)
            segment_label.append(label)
        # Aplicar el desplazamiento temporal al conjunto de muestras
        shift_amount = 0.25 * self.sample_rate_target  # Calcular el desplazamiento en muestras
        shifted_segment = []
        for i in range(10):
            # Calcular el índice de inicio y fin del desplazamiento
            start_index = int((i * shift_amount) % (10 * self.sample_rate_target))
            end_index = int(((i + 1) * shift_amount) % (10 * self.sample_rate_target))
            
            # Recortar y desplazar la muestra actual
            shifted_signal = torch.cat((segment_signal[i][:, start_index:], segment_signal[i][:, :start_index]), dim=1)
            shifted_segment.append(shifted_signal)
        
        # Concatenar las muestras desplazadas en un tensor
        shifted_segment_tensor = torch.cat(shifted_segment, dim=1)
        
        return shifted_segment_tensor, label, filename
    
    
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
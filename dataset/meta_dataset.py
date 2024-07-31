import torch
import pandas as pd
import torchaudio
from matplotlib import pyplot as plt
from dataset.base_dataset import Base_dataset
import warnings


class Meta_dataset(Base_dataset):
    def __init__(self, content_file, audio_dir, transformations , sample_rate_target, device, label_encoder, tensorboard=False):
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
        super().__init__(content_file, audio_dir, transformations , sample_rate_target, device, label_encoder, tensorboard=tensorboard)
        self.content = pd.read_csv(content_file, sep='\t')
        self.audio_dir = audio_dir
        self.device = device
        self.transformations = transformations.to(self.device)
        self.sample_rate_target = sample_rate_target
        self.label_encoder = label_encoder
        self.encoded_labels = self.content.copy()
        self.encoded_labels.iloc[:, 1] = self.label_encoder.fit_transform(self.content.iloc[:, 1])
        

    
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
        
    ### Tensorboard functions ### 
    def _save_class_samples_tensorboard(self):
        filenames = self.content.groupby('scene_label').head(10)
        filenames = filenames.iloc[:, 0]
        print("Saving class samples to tensorboard")
        for filename in filenames:
            index = self.get_index_from_filename(filename)
            audio_sample_path = self._get_audio_sample_path(index)
            filename = self._get_audio_sample_filename(index)
            encoded_label = self._get_audio_sample_label(index)
            identifier = self._get_audio_sample_identifier(index)
            signal, sr = torchaudio.load(audio_sample_path)
            signal = signal.to(self.device)
            #resample and mixdown if necessary (assuming dissonance in the dataset)
            signal = self._resample_if_needed(signal, sr)
            signal = self._mix_down_if_needed(signal)

            self._save_audio_to_tensorboard(signal, encoded_label, filename, identifier)
            signal = self.transformations(signal)
            self._save_mel_spectrogram_to_tensorboard(signal, encoded_label, filename, identifier)

            self.writer.flush()

            self.writer.close()
    
    def get_index_from_filename(self, filename):
        return self.content.index[self.content['filename'] == filename].tolist()[0]

    def _save_audio_to_tensorboard(self, audio, encoded_label, filename, identifier):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.writer.add_audio(f'{encoded_label}/{identifier}/{filename}', audio, sample_rate=self.sample_rate_target)

    def _save_mel_spectrogram_to_tensorboard(self, mel_spectrogram, encoded_label, filename, identifier):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.figure()
            plt.imshow(torch.log(mel_spectrogram[0].cpu() + 1e-9).numpy(), cmap='viridis', aspect='auto', origin='lower')
            plt.title(f'Mel_Spectrogram {encoded_label}/{identifier}/{filename}')
            plt.xlabel('Time')
            plt.ylabel('Mel Filter')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            self.writer.add_figure(f'Mel Spectrogram {encoded_label}/{filename}',  plt.gcf())
            plt.close()
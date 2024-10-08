import torch
import numpy as np 
import os
import torchaudio
#import dataset.base_dataset as base_dataset
from dataset.base_dataset import Base_dataset


class Cached_dataset(Base_dataset):
    def __init__(self, content_file, audio_dir, transformations , transform_probs, sample_rate_target, device, label_encoder, skipCache=[],cache_transforms=True):
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
       
        super().__init__(content_file, audio_dir, None, sample_rate_target, device, label_encoder)
        self.skipCache=skipCache
        if not os.path.exists('data/cache'):
            os.makedirs('data/cache')
        for i,_ in enumerate(transformations):
            if not os.path.exists(f'data/cache/{i}'):
                os.makedirs(f'data/cache/{i}')
        self.transform_sets = {i: transform_set.to(self.device)
                              for i, transform_set in enumerate(transformations)}
        # Calc cumulative sum of transform_probs
        if sum(transform_probs) < 0.99 or sum(transform_probs) > 1.01:
            raise ValueError("The sum of the transform probabilities must be 1")
        self.transform_probs = np.cumsum(transform_probs)
        if len(self.transform_probs) != len(transformations):
            raise ValueError("The number of transform probabilities must be equal to the number of transformations")
        if cache_transforms:
            print("Caching transformations")
            self._cache_transforms()
        #self._test_cached_transforms() 
            
        

    def _transform_data(self, index, transform):
        
        audio_sample_path = self._get_audio_sample_path(index)
        filename = self._get_audio_sample_filename(index)
        device = self._get_audio_recording_device(index)
        signal, sr = torchaudio.load(audio_sample_path)
        


        # Move the signal to the correct device
        signal = signal.to(self.device)

        # Resample and mixdown if necessary (assuming dissonance in the dataset)
        signal = self._resample_if_needed(signal, sr)
        signal = self._mix_down_if_needed(signal)

        # Apply the Compose transformations to the signal
        signal_transformed = transform(signal)
        del signal
        # Move the transformed signal back to the original device
        #signal_transformed = signal_transformed.to('cpu')
        
        return signal_transformed, sr, filename, device

            

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
        for i, transform_set in self.transform_sets.items():
                if i in self.skipCache:
                    print(f"Skipping caching for set {i}")
                else:
                    print(f"Caching transformations for set {i} {type(transform_set)}, {transform_set.transforms[0].__class__.__name__}")
                    if transform_set.transforms[0].__class__.__name__ == 'TimeShiftSpectrogram':
                        print("Caching trasformations for time shifting, this should be a once in a while thing")
                        transform_set.transforms[0].transform_all_clips()
                    else:
                        for j in range(0, len(self.content)):
                            signal, sr, filename, device = self._transform_data(j, transform_set)
                            torch.save(signal, f'data/cache/{i}/{filename}.pt')
                            del signal
        print("Transformations are cached to disk")
    def _get_audio_sample_city(self, index):
        """
        The function `_get_audio_sample_city` is used to get the city of the audio sample at the specified index.
        
        :param index: The `index` parameter is the index of the audio sample
        :return: The city of the audio sample at the specified index
        """
        # From metadata and identifier
        city = self.metadata['identifier'][index].split('-')[0]
        return city
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
        device = self._get_audio_recording_device(index)
        city = self._get_audio_sample_city(index)
        city = self.city_encoder.transform([city])[0]
        while True:
            chosen_transform = self._choose_transform()
            transform_set = self.transform_sets[chosen_transform]
            if hasattr(transform_set.transforms[0], 'sample_random'):
                path = transform_set.transforms[0].sample_random()
                signal = torch.load(path+filename+'.pt')
                return signal, label, filename, device, city
            elif transform_set.transforms[0].__class__.__name__ == 'IRAugmentation':
                #transform back device
                if self.device_encoder.inverse_transform([device])[0] == 'a':
                    signal, sr, filename, device = self._transform_data(index, transform_set)
                    return signal, label, filename, device, city
                else:
                    continue
            elif chosen_transform in self.skipCache:
                    signal, sr, filename, device = self._transform_data(index, transform_set)
                    return signal, label, filename, device, city
            else:
                path = f'data/cache/{chosen_transform}/{filename}.pt'
                signal = torch.load(path)
                return signal, label, filename, device, city
            
    def _choose_transform(self):
        """
        The function `choose_transform` is used to select a random transformation from the list of transformations
        based on the probabilities of each transformation.
        
        :return: The function returns the index of the selected transformation
        """
        return np.argmax(self.transform_probs > np.random.rand())


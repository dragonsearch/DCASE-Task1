from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

import torch.nn.functional as F
import torchaudio
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

class CustomTransformAudio(torch.nn.Module):
    def forward(self, signal):
        #print("Applying custom transform")
        #Apply Vol, TimeMasking, FrequencyMasking, PitchShift at random and then apply mel spectrogram
        rand_num = np.random.rand()
        if rand_num < 0.5:
            signal = Vol(gain=0.5)(signal)
            print("Applied Vol")
        else:
            signal = signal.to("cpu")
            signal = PitchShift(32050, n_steps=4)(signal)
            print("Applied PitchShift")
            signal = signal.to("cuda")
        return signal

class CustomTransformSpectrogram(torch.nn.Module):
    def forward(self, signal):
        # Apply Vol, TimeMasking, FrequencyMasking, TimeStretch
        rand_num = np.random.rand()
        if rand_num < 0.33:
            tm = TimeMasking(time_mask_param=65).to("cuda")
            signal = tm(signal)
            #print("Applied TimeMasking")
        elif rand_num < 0.66:
            # REVISAR RODRIGO
            fm = FrequencyMasking(freq_mask_param=65).to("cuda")
            signal = fm(signal)
            #print("Applied FrequencyMasking")
        else:
            """
            signal = TimeStretch(signal,fixed_rate=0.8)(signal)
            print("Applied TimeStretch")
            """
        return signal

class TimeShiftRange(torch.nn.Module):
    def __init__(self, sample_rate, max_shift):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_shift = max_shift

    def forward(self, signal):
        shift = int(np.random.uniform(-self.max_shift, self.max_shift) * self.sample_rate)
        return torch.roll(signal, shifts=shift, dims=1)

class TimeShiftValue(torch.nn.Module):
    def __init__(self, sample_rate, shift):
        super().__init__()
        self.sample_rate = sample_rate
        self.shift = shift

    def forward(self, signal):
        num_shifts = int(self.shift * self.sample_rate)
        return torch.roll(signal, shifts=num_shifts, dims=1) 

class DIRAugmentation(torch.nn.Module):
    def __init__(self, target_resolution):
        super(DIRAugmentation, self).__init__()
        self.target_resolution = target_resolution

    def forward(self, waveform):
        # Calcular el espectrograma de la forma de onda de entrada
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)

        # Aplicar la DIR augmentation
        # Aquí implementa la lógica para modificar la resolución espectral
        # Puedes utilizar técnicas como la interpolación o el submuestreo
        # para ajustar la resolución del espectrograma

        # Por ejemplo, puedes usar Resample de torchaudio para cambiar la resolución
        resample = Resample(orig_freq=spectrogram.size(1), new_freq=self.target_resolution)
        augmented_spectrogram = resample(spectrogram)

        # Reconstruir la forma de onda a partir del espectrograma modificado
        reconstructed_waveform = torchaudio.transforms.InverseSpectrogram()(augmented_spectrogram)

        return reconstructed_waveform

class IRAugmentation(torch.nn.Module):
    def __init__(self, impulse_response):
        super(IRAugmentation, self).__init__()
        self.impulse_response = impulse_response

    def forward(self, waveform):
        impulse_response = self.impulse_response
        #waveform [1,44100]
        #[1,1,44100]

        # Perform 1D convolution
        
        convolved_waveform = F.conv1d(waveform.unsqueeze(0), impulse_response.unsqueeze(0), padding='same')
        #convolved_waveform = convolved_waveform[:, :convolved_waveform.shape[1]]

        return convolved_waveform.squeeze(0)


# Adjust n_fft and hop_length to be compatible with your signal's length
    """
mel_spectrogram = MelSpectrogram(
    sample_rate=44100,
    n_fft=2048,  # Adjust this value based on your signal
    hop_length=512,  # Adjust this value based on your signal
    n_mels=40,
)

custom_transform = customTransformSpectrogram()
signal, sr = torchaudio.load("data/TAU-urban-acoustic-scenes-2022-mobile-development/audio/street_traffic-prague-1006-41673-0-s5.wav")
signal = mel_spectrogram(signal)

# Show the original mel spectrogram and the modified mel spectrogram side by side
plt.subplot(1, 2, 1)
plt.imshow(signal.cpu().detach().numpy()[0])
plt.subplot(1, 2, 2)
plt.imshow(custom_transform(signal).cpu().detach().numpy()[0])
plt.show()
"""

from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
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

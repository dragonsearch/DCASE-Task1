from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchaudio
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
class log_mel(torch.nn.Module):
    def __init__(self,offset=1e-6, mel_spec=None):
        super().__init__()
        self.offset = offset
        self.mel_spec = mel_spec
        if self.mel_spec is None:
            raise ValueError("Mel spectrogram is required for log_mel transform")

    def forward(self, signal):
        mel_spec = self.mel_spec(signal)
        #mel_spec = AmplitudeToDB()(mel_spec)
        mel_spec = torch.log(mel_spec + self.offset)
        return mel_spec
    
class AugmentMelSTFT(torch.nn.Module):
    def __init__(self, n_mels=256, sr=22050, win_length=520, hopsize=210, n_fft=4096, freqm=0, timem=0,
                 fmin=0.0, fmax=None, fmin_aug_range=1, fmax_aug_range=1):
        """
        :param n_mels: number of mel bins
        :param sr: sampling rate used (same as passed as argument to dataset)
        :param win_length: fft window length in samples
        :param hopsize: fft hop size in samples
        :param n_fft: length of fft
        :param freqm: maximum possible length of mask along frequency dimension
        :param timem: maximum possible length of mask along time dimension
        :param fmin: minimum frequency used
        :param fmax: maximum frequency used
        :param fmin_aug_range: randomly changes min frequency
        :param fmax_aug_range: randomly changes max frequency
        """
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            # fmax is by default set to sampling_rate/2 -> Nyquist!
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hopsize
        # buffers are not trained by the optimizer, persistent=False also avoids adding it to the model's state dict
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        # shape: batch size x samples
        # majority of energy located in lower end of the spectrum, pre-emphasis compensates for the average spectral
        # shape
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)

        # Short-Time Fourier Transform using Hanning window
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        # shape: batch size x freqs (n_fft/2 + 1) x timeframes (samples/hop_length) x 2 (real and imaginary components)

        # calculate power spectrum
        x = (x ** 2).sum(dim=-1)
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        if not self.training:
            # don't augment eval data
            fmin = self.fmin
            fmax = self.fmax

        # create mel filterbank
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        # apply mel filterbank to power spectrogram
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)
        # calculate log mel spectrogram
        melspec = (melspec + 0.00001).log()

        if self.training:
            # don't augment eval data
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization
        return melspec

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




class TimeShiftSpectrogram(torch.nn.Module):

    def __init__(self, sample_rate, shift, metadata, mel_spec):

        super().__init__()

        self.sample_rate = sample_rate

        self.shift = shift

        self.n_shifts = [round(x * 0.1,2) for x in range(1, 10)]

        self.distribution = torch.distributions.uniform.Uniform(1,10)

        self.mel_spec = mel_spec

        self.metadata = metadata

        self.grouped_filenames = self.metadata.groupby(self.metadata['filename'].apply(self.get_groups))

        self.transformed = False

        if not os.path.exists('data/cache/shifted'):

            os.makedirs('data/cache/shifted')

        for n_shift in self.n_shifts:

            if not os.path.exists(f'data/cache/shifted/{n_shift}/audio'):

                os.makedirs(f'data/cache/shifted/{n_shift}/audio')

        self.folder = 'data/cache/shifted'




    def transform_all_clips(self):

        if self.transformed:

            return

        for identifier, group in self.grouped_filenames['filename']:

            torch.cuda.empty_cache() ## EXPERIMENTAL

            signal = self.join_clips(group)

            signal = signal.to('cuda')

            # Shift on 0.1, 0.2, 0.3, 0.4, 0.5

            #shift = int(torch.distributions.uniform(0,1) * self.sample_rate)

            shifts = [int(shift * self.sample_rate) for shift in self.n_shifts]

            for shift_id,shift in zip(self.n_shifts, shifts):

                signal_shifted = torch.roll(signal, shifts=shift, dims=1)

                # Split the signal into sample rate clips

                for i in range(0, len(self.n_shifts)+1):

                    # Apply mel spectrogram

                    mel_spec = self.mel_spec(signal_shifted[:, i:i+self.sample_rate])

                    # Save the mel spectrogram

                    # Switch the filename identifier

                    identifier_copy = identifier.split('-')

                    dev = identifier_copy[-1:][0]

                    identifier_copy = '-'.join(identifier_copy[:-1])

                    filename = f'data/cache/shifted/{shift_id}/{identifier_copy}-{i}-{dev}.wav.pt'

                    torch.save(mel_spec, filename)

        self.transformed = True







    def join_clips(self, group):

        # Load the filenames and join the clips

        signals = []

        for filename in group:

            signal, sr = torchaudio.load('data/TAU-urban-acoustic-scenes-2022-mobile-development/' + filename)

            signals.append(signal)

        # Join the signals

        signal = torch.cat(signals, dim=1)

        return signal

    def sample_random(self):

        n = round(int(self.distribution.sample()) * self.shift,2)

        path = f'data/cache/shifted/{n}/audio/'

        return path

    def get_groups(self, filename):

        #Remove the extension

        filename = filename[:-4]

        # Save the device (separated by -)

        dev = filename.split('-')[-1:]

        filename_switched = filename.split('-')

        # Switch up the last two parts of the filename

        filename = filename_switched[:-2] + filename_switched[-1:]

        # Join the remaining parts

        filename = '-'.join(filename)

        return filename 

    def forward(self, signal):

        # Shift the signal

        if self.transformed:

            return signal

        else:

            self.transform_all_clips()

            return signal







"""

#Create mel_spectrogram

mel_spec = MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=2048, hop_length=1024, n_mels=128)

#Metadata

import pandas as pd

metadata = pd.read_csv('data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', sep='\t')

t = TimeShiftSpectrogram(44100, 0.1, metadata, mel_spec).to('cuda')

#t.transform_all_clips()

"""
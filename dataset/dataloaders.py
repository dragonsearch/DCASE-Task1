import torch
import torchaudio
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from dataset.base_dataset import Base_dataset
from dataset.cached_dataset import Cached_dataset
from dataset.eval_dataset import Eval_dataset
from dataset.meta_dataset import Meta_dataset
from torchvision.transforms import v2
from transforms import CustomTransformSpectrogram, CustomTransformAudio
from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift



def load_dataloaders(trial, params):
    # Load data using the dataloader
    # The data is on /data/TAU-urban-acoustic-scenes-2022-mobile-development/audio folder 
    # and the labels are on /data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv
    # The data is already split into train, and test sets.
    """
    Could be split into more functions later. Just so parameters are not hardcoded
    """
    data_training_path = params['abspath'] + '/data/TAU-urban-acoustic-scenes-2022-mobile-development/'
    data_evaluation_path = params['abspath'] + '/data/TAU-urban-acoustic-scenes-2023-mobile-evaluation/'

    mel_spectrogram, data_augmentation_transforms, data_augmentation_transform_probs = transforms(params)

    audiodataset_train, audiodataset_val, audio_evaluation_dataset, label_encoder = get_dataset(params, 
                 data_training_path,
                 data_evaluation_path, 
                 mel_spectrogram, 
                 data_augmentation_transforms, 
                 data_augmentation_transform_probs)

    test_loader = torch.utils.data.DataLoader(audio_evaluation_dataset, batch_size=params['batch_size'], shuffle=True)

    train_loader = torch.utils.data.DataLoader(audiodataset_train, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(audiodataset_val, batch_size=params['batch_size'], shuffle=True)

    # MNIST dataset for testing
    """
    mnist_dataset = datasets.MNIST(root="mnist", train=True, download=True, transform=ToTensor())
    mnist_test_dataset = datasets.MNIST(root="mnist", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=True)
    """
    return train_loader, val_loader, test_loader, label_encoder

def transforms(params):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = params['sample_rate'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length'],
            n_mels=params['n_mels'],
        ).to(params['device'])
    data_augmentation_transforms = [
    v2.Compose([mel_spectrogram]),
    v2.Compose([mel_spectrogram, FrequencyMasking(freq_mask_param=10).to(params['device'])]),
    v2.Compose([mel_spectrogram, TimeMasking(time_mask_param=10).to(params['device'])]),
    v2.Compose([PitchShift(32050, n_steps=4).to(params['device']), mel_spectrogram]),
    v2.Compose([Vol(gain=0.5).to(params['device']), mel_spectrogram]),

    ]

    data_augmentation_transform_probs = [0.6, 0.1, 0.1, 0.1, 0.1]
    return mel_spectrogram, data_augmentation_transforms, data_augmentation_transform_probs

def get_dataset(params, data_training_path,
                 data_evaluation_path, mel_spectrogram, 
                 data_augmentation_transforms, data_augmentation_transform_probs):
    
    if 'tensorboard' in params and params['tensorboard']:
        audiodataset = Meta_dataset(
            data_training_path + 'meta.csv', 
            data_training_path + 'audio', 
            data_augmentation_transforms, params['sample_rate'],
            'cuda',
            label_encoder=LabelEncoder(),
            tensorboard=True
            )
        
    label_encoder = LabelEncoder()
    audiodataset_train = Cached_dataset(
        data_training_path + 'evaluation_setup/fold1_train.csv',
        data_training_path + 'audio',
        data_augmentation_transforms,
        data_augmentation_transform_probs,
        params['sample_rate'],
        'cuda',
        label_encoder= label_encoder,
        cache_transforms=params['cache_transforms'],
        #data_augmentation=params['data_augmentation']
        )
    audiodataset_val = Cached_dataset(
        data_training_path + 'evaluation_setup/fold1_evaluate.csv',
        data_training_path + 'audio',
        [ v2.Compose([mel_spectrogram]) ],
        [1],
        params['sample_rate'],
        'cuda',
        label_encoder=label_encoder,
        cache_transforms=params['cache_transforms'],
        #data_augmentation=params['data_augmentation']
        )
    # Not using the evaluation set for now
    audio_evaluation_dataset = Eval_dataset(
        data_evaluation_path + 'evaluation_setup/fold1_test.csv', 
        data_evaluation_path + 'audio', 
        mel_spectrogram, params['sample_rate'],
        'cuda'
        )
    return audiodataset_train, audiodataset_val, audio_evaluation_dataset, label_encoder
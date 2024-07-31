import torch
import torchaudio
from sklearn.preprocessing import LabelEncoder
from dataset.cached_dataset import Cached_dataset
from dataset.eval_dataset import Eval_dataset
from dataset.meta_dataset import Meta_dataset
from torchvision.transforms import v2
from dataset.transforms import TimeShiftSpectrogram
from torchaudio.transforms import Vol, TimeMasking, FrequencyMasking, PitchShift
from dataset.transforms import IRAugmentation
import pandas as pd


def load_dataloaders(trial, params):
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

    return train_loader, val_loader, test_loader, label_encoder

def transforms(params):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = params['sample_rate'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length'],
            n_mels=params['n_mels'],
            f_max=None,
            f_min=0
        ).to(params['device'])
    metadata = pd.read_csv('data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', sep='\t')
    t = TimeShiftSpectrogram(params['sample_rate'], 0.1, metadata, mel_spectrogram)
    t.to(params['device'])
    if params['create_timeshift']:
        t.transform_all_clips()
    data_augmentation_transforms = [
    v2.Compose([IRAugmentation().to(params['device']), mel_spectrogram]),
    v2.Compose([mel_spectrogram, FrequencyMasking(freq_mask_param=25).to(params['device'])]),
    v2.Compose([mel_spectrogram, TimeMasking(time_mask_param=25).to(params['device'])]),
    v2.Compose([PitchShift(params['sample_rate'], n_steps=2).to(params['device']), mel_spectrogram]),
    v2.Compose([Vol(gain=0.5).to(params['device']), mel_spectrogram]),
    v2.Compose([t]),
    v2.Compose([mel_spectrogram])
    ]
    data_augmentation_transform_probs = params['transform_probs']
    return mel_spectrogram, data_augmentation_transforms, data_augmentation_transform_probs

def get_dataset(params, data_training_path,
                 data_evaluation_path, mel_spectrogram, 
                 data_augmentation_transforms, data_augmentation_transform_probs):
    
    
    skipCache = params['skip_cache_train']
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
        skipCache=skipCache
        #data_augmentation=params['data_augmentation']
        )
    skipCache = params['skip_cache_val']
    audiodataset_val = Cached_dataset(
        data_training_path + 'evaluation_setup/fold1_evaluate.csv',
        data_training_path + 'audio',
        [ v2.Compose([mel_spectrogram]) ],
        [1],
        params['sample_rate'],
        'cuda',
        label_encoder=label_encoder,
        cache_transforms=params['cache_transforms'],
        skipCache=skipCache
        )
    # Not using the evaluation set for now
    audio_evaluation_dataset = Eval_dataset(
        data_evaluation_path + 'evaluation_setup/fold1_test.csv', 
        data_evaluation_path + 'audio', 
        mel_spectrogram, params['sample_rate'],
        'cuda'
        )
    return audiodataset_train, audiodataset_val, audio_evaluation_dataset, label_encoder

def save_class_samples_to_tb(params):
    data_training_path = params['abspath'] + '/data/TAU-urban-acoustic-scenes-2022-mobile-development/'
    _, data_augmentation_transforms, _ = transforms(params)

    if 'save_class_samples_tensorboard' in params and params['save_class_samples_tensorboard']:
        audiodataset = Meta_dataset(
            data_training_path + 'meta.csv', 
            data_training_path + 'audio', 
            data_augmentation_transforms, params['sample_rate'],
            'cuda',
            label_encoder=LabelEncoder(),
            tensorboard=True
            )
        audiodataset._save_class_samples_tensorboard()
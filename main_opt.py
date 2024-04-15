import torchmetrics
import torchinfo
import torch
import torchaudio
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)

import Evaluator
import numpy as np
import nessi

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

#REMOVE LATER TESTING PURPOSES
from torchvision import datasets
from torchvision.transforms import ToTensor, v2
from torch.utils.data import DataLoader

import dataset
from dataset.base_dataset import Base_dataset
from dataset.cached_dataset import Cached_dataset
from dataset.eval_dataset import Eval_dataset
from dataset.meta_dataset import Meta_dataset

from custom_transforms import CustomTransformSpectrogram, CustomTransformAudio
from torchaudio.transforms import Resample, Vol, TimeMasking, FrequencyMasking, TimeStretch, PitchShift
import optuna
import importlib

# Absolute paths
import os
from Trainer import Trainer, TrainerMixUp
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

    # Val Train split
    #train = int(params['train_split'] * len(audiodataset))
    #val = len(audiodataset) - train
    #train_data, val_data = torch.utils.data.random_split(audiodataset, [train, val])

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

def get_model(params):
    """
    Imports the model from the model_file argument and returns it
    If you wish to use a different model, you can change the model_file argument
    or model_class argument
    """
    # Import the model (from model_path)
    model_file = params['model_file']
    #Import lib
    imp = importlib.import_module(model_file.replace('.py',''))



    model_class = params["model_class"]
    # Create the model
    device = params["device"]
    model = getattr(imp, model_class)(params).to(device)
    return model

def get_optimizer(model, params):
    optimizer_str = params['optimizer']
    lr_str = params['lr']
    optimizer = getattr(torch.optim, optimizer_str)(model.parameters(), lr=lr_str)
    return optimizer

def get_criterion(params):
    loss = params['loss']
    criterion = getattr(torch.nn, loss)()
    return criterion

def get_metrics(params):
    metrics_str = params['metrics']
    metrics = {metric : getattr(torchmetrics.classification, metric)(*metrics_str[metric]) for metric in metrics_str}
    return metrics
def get_scheduler(optimizer, params):
    # Return cosine annealing warm  scheduler
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6, last_epoch=-1)
def objective(trial, params):
    params_copy = params.copy()
    sample_rate = 44100
    n_fft = int(sample_rate * 0.04)
    hop_length = n_fft // 2
    n_mels = 40
    trial_model_params = {
        'batch_size': 258,#trial.suggest_categorical('batch_size', [16,32, 64, 128]),
        'name': trial.suggest_categorical('exp_name', ["TFSEPCONV_mixup_mixstyle_test_transf"]) + str(trial.number),
        'end_epoch': trial.suggest_categorical('end_epoch', [2, 3]),
        "start_epoch": 1,
        "end_epoch": 400,
        'lr' : trial.suggest_float('lr', 1e-3, 1e-2, log=True),
        'mixup_alpha': trial.suggest_categorical('mixup_alpha', [0.3]),
        'mixup_prob': trial.suggest_categorical('mixup_prob', [0.5]),
        'optimizer': "Adam",
        "loss": "CrossEntropyLoss",
        'metrics': {'MulticlassAccuracy': [10,1,'macro'], 'MulticlassConfusionMatrix': [10]},
        'device': "cuda",
        'model_file': 'model_classes.transformers.py',
        "model_class": "BaselineViT",
        "early_stopping_patience": 200,
        "early_stopping_threshold": 0.01,
        "seed": 42,
        "train_split": 0.8,
        # Mel spectrogram parameters
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "cache_transforms": False,
        # Model parameters
        "dropout": 0.5,
        #Tensorboard
        "tensorboard": False

    }
    params_copy.update(trial_model_params)
    torch.device(params_copy['device'])
    torch.manual_seed(params_copy['seed'])
    np.random.seed(params_copy['seed'])
    train_loader, val_loader, _, label_encoder = load_dataloaders(trial, params_copy)
    model = get_model(params_copy)
    optimizer = get_optimizer(model, params_copy)
    criterion = get_criterion(params_copy)
    metrics = get_metrics(params_copy)
    scheduler = get_scheduler(optimizer, params_copy)
    trial_model_params = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'metrics': metrics,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'label_encoder': label_encoder,
        'lr_scheduler': scheduler
    }
    params_copy.update(trial_model_params)
    if 'summary' in params_copy and params_copy['summary']:
        torchinfo.summary(model, input_size=(258, 1,128,44))
    if 'nessi' in params_copy and params_copy['nessi']:
        nessi.get_model_size(model,'torch', input_size=(params_copy['batch_size'],1, 64,44))
    if 'mixup_alpha' in params_copy and 'mixup_prob' in params_copy:
        trainer = TrainerMixUp(params_copy)
    else:
        trainer = Trainer(params_copy)
    loss_dict, metrics_dict = trainer.train()
    #discard -1 values
    loss_dict['val'] = {k: v for k, v in loss_dict['val'].items() if v != -100}
    return loss_dict['val'][max(loss_dict['val'].keys())]


#
# General config
#
do_training = True
do_eval = False
# Create the training loop
if do_training:

    params = {
        'abspath': os.path.abspath('.'),
        #"eval_file" : os.path.abspath("resnet18/ckpt/model_resnet18_1.pth"),
        'summary': False,
        'nessi': False
    }
    #Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, params), n_trials=1)
    #print('Best hyperparameters found were: ', results.get_best_result().config)
    import pickle
    with open('models/study.pkl', 'wb') as f:
        pickle.dump(study, f)


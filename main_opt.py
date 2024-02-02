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

from dataset import AudioDataset_fold, AudioDatasetEval, AudioDataset_with_tensorboard
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

#REMOVE LATER TESTING PURPOSES
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


import optuna


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
            sample_rate = 22050,
            n_fft=2048,
            hop_length=512,
            n_mels=64
        )
    """
    audiodataset = AudioDataset_with_tensorboard(
        data_training_path + 'meta.csv', 
        data_training_path + 'audio', 
        mel_spectrogram, 22050,
        'cuda',
        label_encoder=LabelEncoder()
        )
    """
    audiodataset = AudioDataset_fold(
        data_training_path + 'evaluation_setup/fold1_train.csv',
        data_training_path + 'audio',
        mel_spectrogram, 22050,
        'cuda',
        label_encoder=LabelEncoder()
        )
    
    audio_evaluation_dataset = AudioDatasetEval(
        data_evaluation_path + 'evaluation_setup/fold1_test.csv', 
        data_evaluation_path + 'audio', 
        mel_spectrogram, 22050,
        'cuda'
        )

    # Val Train split
    train = int(0.8 * len(audiodataset))
    val = len(audiodataset) - train
    train_data, val_data = torch.utils.data.random_split(audiodataset, [train, val])

    test_loader = torch.utils.data.DataLoader(audio_evaluation_dataset, batch_size=params['batch_size'], shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)

    # MNIST dataset for testing
    """
    mnist_dataset = datasets.MNIST(root="mnist", train=True, download=True, transform=ToTensor())
    mnist_test_dataset = datasets.MNIST(root="mnist", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=True)
    """
    return train_loader, val_loader, test_loader

def get_model(params):
    """
    Imports the model from the model_file argument and returns it
    If you wish to use a different model, you can change the model_file argument
    or model_class argument
    """
    # Import the model (from model_path)
    model_file = params['model_file']
    imp = __import__(model_file[:model_file.index(".")])

    model_class = params["model_class"]
    # Create the model
    device = params["device"]
    model = getattr(imp, model_class)().to(device)
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

def objective(trial, params):
    params_copy = params.copy()
    trial_model_params = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'name': trial.suggest_categorical('exp_name', ["OptTest"]) + str(trial.number),
        'end_epoch': trial.suggest_categorical('end_epoch', [2, 3]),
        "start_epoch": 1,
        "end_epoch": 100,
        'lr' : trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        #'mixup_alpha': trial.suggest_categorical('mixup_alpha', [0]),
        #'mixup_prob': trial.suggest_categorical('mixup_prob', [0]),
        'optimizer': "Adam",
        "loss": "CrossEntropyLoss",
        'metrics': {'MulticlassAccuracy': [10,1,'macro']},
        'device': "cuda",
        'model_file': 'model.py',
        "model_class": "BaselineDCASECNN",
        "early_stopping_patience": 10,
        "early_stopping_threshold": 0.01,
        "label_encoder": LabelEncoder,
        "seed": 42,
    }
    params_copy.update(trial_model_params)
    torch.device(params_copy['device'])
    torch.manual_seed(params_copy['seed'])
    np.random.seed(params_copy['seed'])

    train_loader, val_loader, _ = load_dataloaders(trial, params_copy)
    model = get_model(params_copy)
    optimizer = get_optimizer(model, params_copy)
    criterion = get_criterion(params_copy)
    metrics = get_metrics(params_copy)
    trial_model_params = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'metrics': metrics,
        'train_loader': train_loader,
        'val_loader': val_loader,
    }

    params_copy.update(trial_model_params)
    if 'summary' in params_copy and params_copy['summary']:
        torchinfo.summary(model, input_size=(params_copy['batch_size'],1, 64,44))
    if 'nessi' in params_copy and params_copy['nessi']:
        nessi.get_model_size(model,'torch', input_size=(params_copy['batch_size'],1, 64,44))
    if 'mixup_alpha' in params_copy and 'mixup_prob' in params_copy:
        trainer = TrainerMixUp(params_copy)
    else:
        trainer = Trainer(params_copy)
    loss_dict, metrics_dict = trainer.train()

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
    study.optimize(lambda trial: objective(trial, params), n_trials=100)
    #print('Best hyperparameters found were: ', results.get_best_result().config)



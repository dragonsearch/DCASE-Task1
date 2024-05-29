import torchmetrics
import torch
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import numpy as np
import importlib
from devAccuracy import DevAccuracy
# Absolute paths

def first_hparams(trial):
    sample_rate = 32000
    n_fft = 2048
    hop_length = 744
    n_mels = 128
    trial_model_params = {
        'batch_size': 516,#trial.suggest_categorical('batch_size', [16,32, 64, 128]),
        'name': trial.suggest_categorical('exp_name', ["TFSEPNET_less_bn_relu_[0.2, 0.2, 0.1, 0, 0.1, 0.2, 0.2]"]) + str(trial.number), 
        'end_epoch': trial.suggest_categorical('end_epoch', [2, 3]),
        "start_epoch": 250,
        "end_epoch": 400,
        'lr' : 1e-3,#trial.suggest_float('lr', 2*1e-5, 3*1e-5, log=True),
        'mixup_alpha': trial.suggest_categorical('mixup_alpha', [0.2]),
        'mixup_prob': trial.suggest_categorical('mixup_prob', [0.5]),
        'optimizer': "AdamW",
        "loss": "CrossEntropyLoss",
        'metrics': {'MulticlassAccuracy': [10,1,'macro'], 'MulticlassConfusionMatrix': [10]},
        'device': "cuda",
        'model_file': 'model_classes.tfsepnet.py',
        "model_class": "TfSepNet",
        "early_stopping_patience": 200,
        "early_stopping_threshold": 0.01,
        "seed": 200,
        "train_split": 0.8,
        # Mel spectrogram parameters
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "cache_transforms": False,
        # Model parameters
        "dropout": 0.4,
        #Tensorboard
        "tensorboard": False,
        "skip_cache_train": [0,1,2],
        "skip_cache_val": [0],
        "transform_probs": [0.2, 0.2, 0.1, 0, 0.1, 0.2, 0.2]


    }
    return trial_model_params


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
    #Cyclical learning rate
    #return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-5, step_size_up=2000, step_size_down=2000, cycle_momentum=False)
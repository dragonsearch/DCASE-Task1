import torchmetrics
import torchinfo
import torch
import torchaudio
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
from Trainer import Trainer
import Evaluator
import numpy as np
import nessi

from dataset import AudioDataset, AudioDatasetEval
from model import BasicCNNNetwork

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

#REMOVE LATER TESTING PURPOSES
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


import optuna


# Absolute paths
import os
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
        'lr' : trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        'mixup_alpha': trial.suggest_categorical('mixup_alpha', [0]),
        'mixup_prob': trial.suggest_categorical('mixup_prob', [0]),
        'optimizer': "Adam",
        "loss": "CrossEntropyLoss",
        'metrics': {'MulticlassAccuracy': [10,1,'macro']},
        'device': "cuda",
        'model_file': 'model.py',
        "model_class": "BasicCNNNetwork",
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

    if 'summary' in params_copy and params_copy['summary']:
        torchinfo.summary(model, input_size=(params_copy['batch_size'],1, 64,44))
    if 'nessi' in params_copy and params_copy['nessi']:
        nessi.get_model_size(model,'torch', input_size=(params_copy['batch_size'],1, 64,44))
    if 'mixup_alpha' in params_copy and 'mixup_prob' in params_copy:
        from Trainer import TrainerMixUp as Trainer

    trial_model_params = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'metrics': metrics,
        'train_loader': train_loader,
        'val_loader': val_loader,
    }

    params_copy.update(trial_model_params)
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


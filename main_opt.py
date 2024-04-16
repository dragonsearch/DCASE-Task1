import torchinfo
import torch
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)

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
from devAccuracy import DevAccuracy
from cityAccuracy import CityAccuracy
# Absolute paths
import os
from Trainer import Trainer
from trainer_mixup import TrainerMixUp
from hparams import first_hparams, get_model, get_optimizer, get_criterion, get_metrics, get_scheduler
from dataset.dataloaders import load_dataloaders
def objective(trial, params):
    params_copy = params.copy()
    trial_model_params = first_hparams(trial)
    params_copy.update(trial_model_params)
    torch.device(params_copy['device'])
    torch.manual_seed(params_copy['seed'])
    np.random.seed(params_copy['seed'])
    train_loader, val_loader, _, label_encoder = load_dataloaders(trial, params_copy)
    model = get_model(params_copy)
    optimizer = get_optimizer(model, params_copy)
    criterion = get_criterion(params_copy)
    metrics = get_metrics(params_copy)
    metrics['DevAccuracy'] = DevAccuracy(num_devices=9)
    metrics['CityAccuracy'] = CityAccuracy(num_cities=10)
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
        'summary': True,
        'nessi': False
    }
    #Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, params), n_trials=1)
    #print('Best hyperparameters found were: ', results.get_best_result().config)
    import pickle
    with open('models/study.pkl', 'wb') as f:
        pickle.dump(study, f)


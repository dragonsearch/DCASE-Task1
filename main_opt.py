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


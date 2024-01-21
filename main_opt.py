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

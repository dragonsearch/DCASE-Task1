import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import numpy as np

import pathlib
import pickle
from utils import load_ckpt, save_ckpt
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import time
from utils import save_obj, save_ckpt, load_ckpt


class evaluation():
    """
    Class for evaluating the model on the test set.
    This class assumes that the model is already trained and that we don't the truth labels.
    Output is a dictionary with the probabilities of each class for each sample in the format:
    {filename: [0.2, ..., n_classes], filename: [0.2, ..., n_classes], ...}

    Requires that the test_loader has returns the filename of each sample on the getitem method
    in the following format: (sample, label, filename)

    TODO: Testing
    """
    def __init__(self, model, test_loader, name) -> None:
        self.model = model
        self.test_loader = test_loader
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_total_steps_test = len(self.test_loader)
        self.predictions = {} # File: probabilities of each class for each sample

    def save_preds(self):
        save_obj(self.predictions, self.name +"/plots/preds" + "_" + str(self.name))

    def eval_batches(self):
        for i, (samples, labels, filenames) in enumerate(self.test_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            y_pred = self.model(samples)
            self.predictions.update({filename: y_pred[j].cpu().detach().numpy() for j, filename in enumerate(filenames)})

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            self.eval_batches()
            self.save_preds()
            self.model.train()
        print("Evaluation done")





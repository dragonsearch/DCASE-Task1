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
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard
import time
from utils import save_obj, save_ckpt, load_ckpt, load_obj, dict_to_txt
import os
import shutil
from Trainer import Trainer, EarlyStoppingException

class TrainerMixUp(Trainer):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.mixup_alpha = params['mixup_alpha']
        self.mixup_prob = params['mixup_prob'] if 'mixup_prob' in params else 0
    
    def mixUpCriterion(self, y_pred, y_a, y_b, lam):
        # There are 2 ways of doing this. One for onehot encoded labels and one 
        # for non onehot encoded labels.
        # For onehot encoded labels (as in original paper):
        # loss = lam * self.criterion(y_pred, y_true) + (1 - lam) * self.criterion(y_pred, y_true)
        # For non onehot encoded labels (as in the original implementation):
        # loss = lam * self.criterion(y_pred, y_a) + (1 - lam) * self.criterion(y_pred, y_b)
        # Notice lerp makes us swap the order of the parameters
        loss = torch.lerp(self.criterion(y_pred, y_b), self.criterion(y_pred, y_a), lam)

        return loss
    
    def train_step(self, samples, labels_a, labels_b, lam):
        # Forward pass
        self.optimizer.zero_grad()
        y_pred = self.model(samples)
        loss = self.mixUpCriterion(y_pred, labels_a, labels_b, lam)
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        return y_pred, loss
    
    def mixup_data(self, x, y):
        # Similar to the original implementation  
        # Remember giving alpha = 0 is the same as not using mixup
        if self.mixup_alpha > 0:
            lam =  np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        if self.mixup_prob > 0:
            if np.random.rand() > self.mixup_prob:
                lam = 1
        index = torch.randperm(x.size(0)).to(self.device)
        #mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_x = torch.lerp(x[index, :], x, lam)
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self,epoch):
        """
        Trains the model for each epoch
        """
        time_epoch = time.time() 
        print(f"Epoch {epoch}/{self.num_epochs}")

        for i, (samples, labels, *rest) in enumerate(self.train_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            samples, labels_a, labels_b, lam = self.mixup_data(samples, labels)
            y_pred,loss = self.train_step(samples, labels_a, labels_b , lam)
            self.lr_scheduler.step(epoch + i / self.n_total_steps_train)
            # Add loss and metrics
            self.loss_dict["train"][epoch] += loss.item()
            devices = rest[1].to(self.device)
            self.add_to_metric(y_pred, labels)
            if 'DevAccuracy' in self.metrics:
                self.add_to_dev_accuracy(y_pred, labels, devices)
            #Add scalars to a Tensorboard 
            step = (epoch - 1) * len(self.train_loader) + i
            self.scalars_to_writer(loss, "train (Step)", step)
            if (i+1) % 100 == 0:
                print (f'Step [{i+1}/{self.n_total_steps_train}], Loss: {loss.item():.4f}, Time: {time.time()-time_epoch:.2f} s')
        # Compute metrics
        self.compute_metrics(epoch, val=False)

        # Compute loss
        self.loss_dict["train"][epoch] /= self.n_total_steps_train
        #Add scalars to a Tensorboard for each epoch
        self.scalars_to_writer(loss, "train (Epoch)", step)
        print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Loss: {self.loss_dict['train'][epoch]:.4f}, Time: {time.time()-time_epoch:.2f} s")

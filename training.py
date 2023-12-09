import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import numpy as np
import torchsummary
import pathlib
import pickle
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import time
from utils import save_obj, save_ckpt, load_ckpt, load_obj
import os
import shutil

class Training:
    """
    Warning: metric states are not saved to files and are reset at each epoch, 
    if your metric for some reason has a state worth keeping consider modifying 
    the saving and loading functions as well as the reset_metrics function.
    (This is not the case for most metrics).
    """
    def __init__(self, model, train_loader, val_loader, params) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = params['model']
        self.model.to(self.device)
        self.name = params['name']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = params['criterion']
        self.optimizer = params['optimizer']
        self.start_epoch = params['start_epoch']
        self.num_epochs = params['end_epoch'] - params['start_epoch']
        self.metrics = params['metrics']
        for metrics in self.metrics:
            self.metrics[metrics].to(self.device)
        
        self.n_total_steps_train = len(self.train_loader)
        self.n_total_steps_val = len(self.val_loader)
        self.loss_dict = {stage : {i:0 for i in range(1,self.num_epochs+1)} for stage in ["train", "val"]}
        self.metrics_dict = { stage : {str(metric) : {i:0 for i in range(1,self.num_epochs+1)} }
                             for metric in self.metrics for stage in ["train", "val"]}
        self.prepare_dirs()
        # Resuming training
        if self.start_epoch > 1:
            self.load_dicts()
            self.load_model(self.start_epoch-1)
        
    
    """
    Creates the directories for the plots and the checkpoints.
    """
    def prepare_dirs(self):
        if not os.path.exists('models/' + self.name):
            os.makedirs('models/' + self.name + "/ckpt", exist_ok=True)
            os.makedirs('models/' + self.name + "/plots", exist_ok=True) # Recursivity of makedirs -> exist_ok=True
        else:
            print("Some directory with that name already exists, remove it? (y/n)")
            answer = input()
            if answer == "y":
                shutil.rmtree('models/' + self.name)
                os.makedirs('models/' + self.name + "/ckpt")
                os.makedirs('models/' + self.name + "/plots", exist_ok=True)
            

    """
    Saves the dictionaries of loss and metrics.
    """        
    def save_dicts(self):
        save_obj(self.loss_dict, 'models/' + self.name + "/plots/loss_dict" + "_" + str(self.name))
        save_obj(self.metrics_dict, 'models/' + self.name + "/plots/metrics_dict" + "_" + str(self.name))

    def load_dicts(self):
        self.loss_dict = load_obj('models/' + self.name +"/plots/loss_dict" + "_" + str(self.name))
        self.metrics_dict = load_obj('models/' + self.name + "/plots/metrics_dict" + "_" + str(self.name))

    def load_model(self, epoch):
        ckpt_path = 'models/' + self.name + "/ckpt" + "/model_" + str(self.name) + '_' + str(epoch) + ".pth"
        self.model, self.optimizer = load_ckpt(self.model, self.optimizer, ckpt_path)
        self.model = self.model.to(self.device)
        print("Loading model with loss: ", self.loss_dict["train"][epoch], "from ", ckpt_path)
    
    """
    Saves the model, the optimizer and the dictionaries
    """
    def save_model(self, epoch):
        ckpt_path = 'models/' + self.name + "/ckpt" + "/model_" + str(self.name) + '_' + str(epoch) + ".pth"
        save_ckpt(self.model, self.optimizer, ckpt_path, epoch)
        print("Saving model with loss: ", self.loss_dict["train"][epoch], "as ", ckpt_path)
        self.save_dicts()

    def reset_metrics(self):
        """
        Torchmetrics reset
        """
        for metric in self.metrics:
            self.metrics[metric].reset()


    def add_to_metric(self, y_pred,y_true):
        """
        Adds the predictions and the labels to the metrics
        """
        y_true = torch.as_tensor(y_true, dtype=torch.float64)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float64)

        for metric in self.metrics:
            self.metrics[metric].update(y_pred, y_true)


 
    def train_epoch(self,epoch):
        """
        Trains the model for each epoch
        """
        time_epoch = time.time() 
        print(f"Epoch {epoch}/{self.num_epochs}")

        for i, (samples, labels) in enumerate(self.train_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            y_pred,loss = self.train_step(samples, labels)
            # Add loss and metrics
            self.loss_dict["train"][epoch] += loss.item()
            self.add_to_metric(y_pred, labels)
            if (i+1) % 10 == 0:
                print (f'Step [{i+1}/{self.n_total_steps_train}], Loss: {loss.item():.4f}, Time: {time.time()-time_epoch:.2f} s')

        # Compute metrics
        self.compute_metrics(epoch, val=False)

        # Compute loss
        self.loss_dict["train"][epoch] /= self.n_total_steps_train

        print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Loss: {self.loss_dict['train'][epoch]:.4f}, Time: {time.time()-time_epoch:.2f} s")



    def train_step(self, samples, labels):
        """
        Trains the model for each batch
        """
        # Forward pass
        self.optimizer.zero_grad()
        y_pred = self.model(samples)

        loss = self.criterion(y_pred, labels)

        # Backward and optimize

        loss.backward()
        self.optimizer.step()

        return y_pred, loss


    def val_epoch(self, epoch):
        print( "Validation started")
        self.model.eval()
        with torch.no_grad():
            time_step = time.time()
            print(f"Validation batch")
            for i, (samples, labels) in enumerate(self.val_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                y_pred,loss = self.val_step(samples, labels, epoch)
                # Add loss and metrics
                self.loss_dict["val"][epoch] += loss.item()
                self.add_to_metric(y_pred, labels)
                print (f'Step [{i+1}/{self.n_total_steps_val}], Loss: {loss.item():.4f}, Time: {time.time()-time_step:.2f} s')


            # Compute loss
            self.loss_dict["val"][epoch] /= self.n_total_steps_val
            # Compute metrics
            self.compute_metrics(epoch, val=True)
            
            print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Val Loss: {self.loss_dict['val'][epoch]:.4f}, Time: {time.time()-time_step:.2f} s")
        self.model.train()
        print( "Validation ended")
        
    def val_step(self, samples, labels, epoch):
        with torch.no_grad():
            # Forward pass
            y_pred = self.model(samples)
            loss = self.criterion(y_pred, labels)
            return y_pred, loss
        
    def compute_metrics(self, epoch, val=False):
        #Compute metrics
        for metric in self.metrics:
            if val:
                self.metrics_dict['val'][str(metric)][epoch] = self.metrics[metric].compute()
            else:
                self.metrics_dict['train'][str(metric)][epoch] = self.metrics[metric].compute()



    def train(self):
        self.model.train()
        time_start = time.time()
        for ep in range(self.start_epoch,self.start_epoch + self.num_epochs):
            self.train_epoch(ep)
            self.reset_metrics()
            self.val_epoch(ep)
            self.reset_metrics()
            self.save_model(ep)
        print(f'Finished Training in {time.time()-time_start:.2f} s')
        return self.loss_dict, self.metrics_dict
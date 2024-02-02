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
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard
import time
from utils import save_obj, save_ckpt, load_ckpt, load_obj, dict_to_txt
import os
import shutil

class Trainer():
    """
    Warning: metric states are not saved to files and are reset at each epoch, 
    if your metric for some reason has a state worth keeping consider modifying 
    the saving and loading functions as well as the reset_metrics function.
    (This is not the case for most metrics).
    """
    def __init__(self, params) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = params['model']
        self.model.to(self.device)
        self.name = params['name']
        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']
        self.criterion = params['criterion']
        self.optimizer = params['optimizer']
        self.start_epoch = params['start_epoch']
        self.num_epochs = params['end_epoch'] - params['start_epoch']
        self.metrics = params['metrics']
        self.abspath = os.path.abspath(params['abspath'])
        self.writer = SummaryWriter(f'runs/{self.name}') # Create SummaryWriter object
        self.early_stopping_patience = params['early_stopping_patience']
        self.early_stopping_threshold = params['early_stopping_threshold']
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        for metrics in self.metrics:
            self.metrics[metrics].to(self.device)
        
        self.n_total_steps_train = len(self.train_loader)
        self.n_total_steps_val = len(self.val_loader)
        self.loss_dict = {stage : {i:0 for i in range(1,self.num_epochs+1)} for stage in ["train", "val"]}
        self.metrics_dict = { stage : {str(metric) : {i:0 for i in range(1,self.num_epochs+1)} }
                             for metric in self.metrics for stage in ["train", "val"]}
        self.params = params.copy()
        self.prepare_dirs()
        self.save_exec_params()
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
            print("I'm trying to remove: " + self.abspath + "/models/" + self.name)
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
        self.writer.close()

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

    def save_exec_params(self):
        #save_obj(self.params, 'models/' + self.name + "/plots/exec_params" + "_" + str(self.name))
        dict_to_txt(self.params, 'models/' + self.name + "/plots/exec_params" + "_" + str(self.name) + ".txt")

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
        for i, (samples, labels, *rest) in enumerate(self.train_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            y_pred,loss = self.train_step(samples, labels)
            # Add loss and metrics
            self.loss_dict["train"][epoch] += loss.item()
            self.add_to_metric(y_pred, labels)

            #Add scalars to a Tensorboard
            step = (epoch - 1) * len(self.train_loader) + i
            self.writer.add_scalar('Loss/train', loss.item(), step)
            for metric_name, metric in self.metrics.items():
                self.writer.add_scalar(f'{metric_name}/Train', metric.compute(), step) 
            if (i+1) % 100 == 0:
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
            for i, (samples, labels,*rest) in enumerate(self.val_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                y_pred,loss = self.val_step(samples, labels, epoch)
                # Add loss and metrics
                self.loss_dict["val"][epoch] += loss.item()
                self.add_to_metric(y_pred, labels)
                #Add scalars to Tensorboard
                step = (epoch - 1) * len(self.val_loader) + i
                self.writer.add_scalar('Loss/Val', loss.item(), step)
                for metric_name, metric in self.metrics.items():
                    self.writer.add_scalar(f'{metric_name}/Val', metric.compute(), step)
                if (i+1) % 100 == 0:
                    print (f'Step [{i+1}/{self.n_total_steps_val}], Loss: {loss.item():.4f}, Time: {time.time()-time_step:.2f} s')


            # Compute loss
            self.loss_dict["val"][epoch] /= self.n_total_steps_val
            # Compute metrics
            self.compute_metrics(epoch, val=True)

            print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Val Loss: {self.loss_dict['val'][epoch]:.4f}, Time: {time.time()-time_step:.2f} s")

            # Early stopping
            if self.early_stopping(epoch):
                raise Exception("Early stopping")
            
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

    def early_stopping(self, epoch):
            # Calcular la pérdida promedio en el conjunto de validación
            val_loss = self.loss_dict["val"][epoch]

            # Verificar si la pérdida actual es mejor que la mejor pérdida registrada
            if val_loss + self.early_stopping_threshold < self.best_val_loss :
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Detener el entrenamiento si no hay mejora después de cierta cantidad de épocas
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f'Early stopping at epoch {epoch} due to no improvement in validation loss.')
                return True  # Indica que el entrenamiento debe detenerse
            else:
                return False

    def train(self):
        self.model.train()
        time_start = time.time()
        for ep in range(self.start_epoch,self.start_epoch + self.num_epochs):
            self.train_epoch(ep)
            self.reset_metrics()
            try:
                self.val_epoch(ep)
            except Exception as e:
                print(e)
                break
            self.reset_metrics()
            self.save_model(ep)
        print(f'Finished Training in {time.time()-time_start:.2f} s')
        return self.loss_dict, self.metrics_dict
    
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
            # Add loss and metrics
            self.loss_dict["train"][epoch] += loss.item()
            self.add_to_metric(y_pred, labels)
            if (i+1) % 100 == 0:
                print (f'Step [{i+1}/{self.n_total_steps_train}], Loss: {loss.item():.4f}, Time: {time.time()-time_epoch:.2f} s')
        # Compute metrics
        self.compute_metrics(epoch, val=False)

        # Compute loss
        self.loss_dict["train"][epoch] /= self.n_total_steps_train

        print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Loss: {self.loss_dict['train'][epoch]:.4f}, Time: {time.time()-time_epoch:.2f} s")

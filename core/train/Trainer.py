import torch as torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard
import time
from tools.utils import save_obj, save_ckpt, load_ckpt, load_obj, dict_to_txt
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
        
        self.label_encoder = params['label_encoder']   
        self.n_total_steps_train = len(self.train_loader)
        self.n_total_steps_val = len(self.val_loader)
        self.loss_dict = {stage : {i:-100 for i in range(1,self.num_epochs+1)} for stage in ["train", "val"]}
        self.metrics_dict = {stage : None for stage in ["train", "val"]}
        self.lr_scheduler = params['lr_scheduler']
        for stage in ["train", "val"]:
            self.metrics_dict[stage] = {str(metric) : {i:-100 for i in range(1,self.num_epochs+1)} for metric in self.metrics} 
        self.params = params.copy()
        self.prepare_dirs()
        self.save_exec_params()
        # Resuming training
        if self.start_epoch > 1:
            self.load_dicts()
            self.load_model(self.start_epoch-1)
        
        #Save the parameters and metrics to tensorboard
        text = ["batch_size", "criterion", "device", "dropout", "early_stopping_patience","early_stopping_threshold","end_epoch", "hop_length","loss","lr",
                "model_class","n_fft","n_mels","optimizer","sample_rate","seed","start_epoch","train_split","test_aplit","name","metrics","nessi","summary"]
        for params, value in self.params.items():
            if params in text:
                self.writer.add_text(params, str(value))
        for metrics, value in self.metrics.items():
            self.writer.add_text(metrics, str(value))

        #Save model graph to tensorboard with a example input from the dataloader
        #self.writer.add_graph(self.model, next(iter(self.train_loader))[0].to(self.device))
    
        
    
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
        #lr_sc_dummy = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        #optimizer_dummy = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model, self.optimizer = load_ckpt(self.model, self.optimizer, self.lr_scheduler, ckpt_path)
        #self.model, _ = load_ckpt(self.model, optimizer_dummy, lr_sc_dummy, ckpt_path)
        self.model = self.model.to(self.device)
        print("Loading model with loss: ", self.loss_dict["train"][epoch], "from ", ckpt_path)
    
    """
    Saves the model, the optimizer and the dictionaries
    """
    def save_model(self, epoch):
        ckpt_path = 'models/' + self.name + "/ckpt" + "/model_" + str(self.name) + '_' + str(epoch) + ".pth"
        save_ckpt(self.model, self.optimizer, self.lr_scheduler, ckpt_path, epoch)
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
            if metric != "DevAccuracy" and metric != "CityAccuracy":
                self.metrics[metric].update(y_pred, y_true)
                

    def add_to_dev_accuracy(self, y_pred, y_true, devices):
        """
        Adds the predictions and the labels to the metrics
        """
        self.metrics['DevAccuracy'].update(y_pred, y_true, devices)
 
    def train_epoch(self,epoch):
        """
        Trains the model for each epoch
        """
        time_epoch = time.time() 

        
        
        print(f"Epoch {epoch}/{self.num_epochs}")
        self.loss_dict["train"][epoch] = 0
        for i, (samples, labels, *rest) in enumerate(self.train_loader):
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            y_pred,loss = self.train_step(samples, labels)
            devices = rest[1].to(self.device)
            self.lr_scheduler.step(epoch + i / self.n_total_steps_train)
            # Add loss and metrics
            self.loss_dict["train"][epoch] += loss.item()
            self.add_to_metric(y_pred, labels)
            if 'DevAccuracy' in self.metrics:
                self.add_to_dev_accuracy(y_pred, labels, devices)
            if 'CityAccuracy' in self.metrics:
                cities = rest[2].to(self.device)
                self.add_to_city_accuracy(y_pred, labels, cities)
            #Add scalars to a Tensorboard 
            step = (epoch - 1) * len(self.train_loader) + i
            self.scalars_to_writer(loss, "train (Step)", step)

            if (i+1) % 100 == 0:
                print (f'Step [{i+1}/{self.n_total_steps_train}], Loss: {loss.item():.4f}, Time: {time.time()-time_epoch:.2f} s')
        
        
        # Compute metrics
        self.compute_metrics(epoch, val=False)

        # Compute loss
        self.loss_dict["train"][epoch] /= self.n_total_steps_train

        print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Loss: {self.loss_dict['train'][epoch]:.4f}, Time: {time.time()-time_epoch:.2f} s")

        #Add scalars to a Tensorboard for each epoch
        self.scalars_to_writer(loss, "train (Epoch)", step)
    
    def scalars_to_writer(self, loss, name, step):
        self.writer.add_scalar(f'Loss/{name}', loss.item(), step)
        for metric_name, metric in self.metrics.items():
            if metric_name != "MulticlassConfusionMatrix" and metric_name != "DevAccuracy" and metric_name != "CityAccuracy":
                self.writer.add_scalar(f'{metric_name}/{name}', metric.compute(), step)

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
            self.loss_dict["val"][epoch] = 0
            for i, (samples, labels,*rest) in enumerate(self.val_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                devices = rest[1].to(self.device)
                y_pred,loss = self.val_step(samples, labels, epoch)
                # Add loss and metrics
                self.loss_dict["val"][epoch] += loss.item()
                self.add_to_metric(y_pred, labels)
                if 'DevAccuracy' in self.metrics:
                    self.add_to_dev_accuracy(y_pred, labels, devices)
                if 'CityAccuracy' in self.metrics:
                    cities = rest[2].to(self.device)
                    self.add_to_city_accuracy(y_pred, labels, cities)
                #Add scalars to Tensorboard
                step = (epoch - 1) * len(self.val_loader) + i
                self.scalars_to_writer(loss, "val (Step)", step)
                if (i+1) % 100 == 0:
                    print (f'Step [{i+1}/{self.n_total_steps_val}], Loss: {loss.item():.4f}, Time: {time.time()-time_step:.2f} s')


            # Compute loss
            self.loss_dict["val"][epoch] /= self.n_total_steps_val
            # Compute metrics
            self.compute_metrics(epoch, val=True)

            print(f"Epoch {epoch}/{self.start_epoch + self.num_epochs-1}, Val Loss: {self.loss_dict['val'][epoch]:.4f}, Time: {time.time()-time_step:.2f} s")  
            
            #Print device accuracy
            if 'DevAccuracy' in self.metrics_dict['val']:
                self.plot_dev_accuracy(epoch)
            if 'MulticlassConfusionMatrix' in self.metrics_dict['val']:
                self.confusion_matrix(epoch)
            if 'CityAccuracy' in self.metrics_dict['val']:
                self.plot_city_accuracy(epoch)  
                print(self.metrics_dict['val']['CityAccuracy'][epoch].cpu().numpy())
            #Plots for every epoch
            self.scalars_to_writer(loss, "val (Epoch)", step)
            # Early stopping
            if self.early_stopping(epoch):
                raise EarlyStoppingException()
        self.model.train()
        print( "Validation ended")
    
    def add_to_city_accuracy(self, y_pred, y_true, cities):
        """
        Adds the predictions and the labels to the metrics
        """
        self.metrics['CityAccuracy'].update(y_pred, y_true, cities)

    def plot_city_accuracy(self, epoch):
        fig, ax = plt.subplots(figsize=(18,18))
        city_accuracy = self.metrics_dict['val']['CityAccuracy'][epoch].cpu().numpy()
        ax.bar(np.arange(len(city_accuracy)), city_accuracy)
        ax.set_xticks(np.arange(len(city_accuracy)))
        ax.set_xticklabels(self.val_loader.dataset.city_encoder.inverse_transform(np.arange(len(city_accuracy))))
        city_accuracy_image = ax.figure
        city_accuracy_image_bytes = self.plot_to_image(city_accuracy_image)
        self.writer.add_image(f"City Accuracy/Epoch{epoch}", city_accuracy_image_bytes, epoch)
        plt.close()
        print(self.metrics_dict['val']['CityAccuracy'][epoch].cpu().numpy())

    def plot_dev_accuracy(self, epoch):

        fig, ax = plt.subplots(figsize=(18,18))
        device_accuracy = self.metrics_dict['val']['DevAccuracy'][epoch].cpu().numpy()
        ax.bar(np.arange(len(device_accuracy)), device_accuracy)
        ax.set_xticks(np.arange(len(device_accuracy)))
        ax.set_xticklabels(self.val_loader.dataset.device_encoder.inverse_transform(np.arange(len(device_accuracy))))
        device_accuracy_image = ax.figure
        device_accuracy_image_bytes = self.plot_to_image(device_accuracy_image)
        self.writer.add_image(f"Device Accuracy/Epoch{epoch}", device_accuracy_image_bytes, epoch)
        plt.close()
        print(self.metrics_dict['val']['DevAccuracy'][epoch].cpu().numpy())

    def confusion_matrix(self, epoch):
        # 0-9- > classes
        #Display a big confusion matrix
        fig, ax = plt.subplots(figsize=(18,18))
        conf_matrix = self.metrics_dict['val']['MulticlassConfusionMatrix'][epoch].cpu().numpy()
        # Confusion matrix inverse transform labels
        class_labels = self.label_encoder.inverse_transform(np.arange(conf_matrix.shape[0]))
        confusion_display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_labels)
        confusion_image = confusion_display.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f").figure_
        confusion_image_bytes = self.plot_to_image(confusion_image)
        #Add to tensorboard
        self.writer.add_image(f"Confusion Matrix/Epoch{epoch}", confusion_image_bytes, epoch)
        plt.close()
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

    def plot_to_image(self, figure):
        buf = BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)

        tensor_image = F.to_tensor(image)
        tensor_image = tensor_image / tensor_image.max()

        return tensor_image
    def train(self):
        self.model.train()
        time_start = time.time()
        for ep in range(self.start_epoch,self.start_epoch + self.num_epochs):
            self.train_epoch(ep)
            self.reset_metrics()
            try:
                self.val_epoch(ep)
            except EarlyStoppingException as e:
                print(e)
                break
            self.reset_metrics()
            self.save_model(ep)
        print(f'Finished Training in {time.time()-time_start:.2f} s')
        return self.loss_dict, self.metrics_dict

class EarlyStoppingException(Exception):
    # Exception raised for early stopping

    def __init__(self, message="Early stopping"):
        self.message = message
        super().__init__(self.message)
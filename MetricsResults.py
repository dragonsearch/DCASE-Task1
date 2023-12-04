import numpy as np
import pickle
from utils import load_obj
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
class MetricResults:
    dict_metrics = None
    epochs = None
    def __init__(self,name):
        self.name = name
        self.dict_metrics = load_obj("models/" + self.name + "/plots/metrics_dict_" + self.name)
        # Move each value to the cpu
        for stage in self.dict_metrics:
            for metric in self.dict_metrics[stage]:
                self.dict_metrics[stage][metric] = {k: v.cpu().detach().numpy()
                                                    if type(v) is torch.Tensor 
                                                    else v
                                                    for k, v in self.dict_metrics[stage][metric].items()}
        self.dict_loss = load_obj("models/" + self.name + "/plots/loss_dict_" + self.name)
    
    def plotAll(self,metrics=None, loss=True):
        if metrics is None:
            metrics = self.dict_metrics['train'].keys() #Train == val
        
        if loss == True:
            self.plotLoss()

        self.plotMetrics(metrics)

    def plotMetrics(self, metrics):
        for metric in metrics:
            plt.figure(figsize=(15, 7))
            plt.subplot(1, 1, 1)
            plt.plot(self.dict_metrics['train'][metric].keys(), self.dict_metrics['train'][metric].values(), label="train_" + metric)
            plt.plot(self.dict_metrics['val'][metric].keys(), self.dict_metrics['val'][metric].values(), label="val_" + metric)
            plt.title(metric)
            plt.xticks(np.arange(1, max(self.dict_metrics['train'][metric].keys())+1, 1.0))
            plt.xlabel(metric)
            plt.legend()
            plt.savefig("models/" + self.name + "/plots/" + metric + ".png")
            print(f'Saved {metric} plot in models/{self.name}/plots/{metric}.png')

    def plotLoss(self):
        loss = list(self.dict_loss["train"].values())
        val_loss = list(self.dict_loss["val"].values())

        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 1, 1)
        plt.plot(self.dict_loss['train'].keys(), loss, label="train_loss")
        plt.plot(self.dict_loss['val'].keys(), val_loss, label="val_loss")
        plt.title("Loss")
        plt.xticks(np.arange(1, max(self.dict_loss['val'].keys()), 1.0))
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig("models/" + self.name + "/plots/loss.png")
        print(f'Saved loss plot in models/{self.name}/plots/loss.png')


if __name__ == "__main__":
    metrics = MetricResults("resnet18")
    metrics.plotAll()
    print('Finished plotting')


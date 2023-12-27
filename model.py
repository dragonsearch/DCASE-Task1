import torch
from torch import nn
class Model(nn.Module):
## Simple NN for testing purposes for fashionMNIST
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        x = self.softmax(logits)
        
        return x

class BasicCNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #4 conv blocks -> flatten -> linear -> softmax
        self.conv1d = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels =16,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels =32,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3d = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels =64,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4d = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels =128,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv1d(x)
        x = self.conv2d(x)
        x = self.conv3d(x)
        x = self.conv4d(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)

        
        return predictions
    



    
import torch
import torch.nn as nn
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
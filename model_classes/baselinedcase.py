import torch
from torch import nn
import random
class BaselineDCASECNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.activation_1 = nn.ReLU()
        
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.activation_2 = nn.ReLU()
        
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.dropout_1 = nn.Dropout(params['dropout'])
        
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.activation_3 = nn.ReLU()
        
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_2 = nn.Dropout(params['dropout'])
        
        self.flatten = nn.Flatten()
        # Ajustamos la dimensión de salida de la capa densa 1
        self.dense_1 = nn.Linear(32*5*11, 100)  # Ajustamos esta dimensión según lo calculado
        self.dropout_3 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(100, 10)

    def forward(self, x):
        print(x.shape)
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.activation_1(x)
        
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.activation_2(x)
        
        x = self.max_pooling_1(x)
        x = self.dropout_1(x)
        
        x = self.conv2d_3(x)
        x = self.batch_norm_3(x)
        x = self.activation_3(x)
        
        x = self.max_pooling_2(x)
        x = self.dropout_2(x)
        
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_3(x)
        x = self.dense_2(x)

        return x
    
class BaselineDCASECNN2(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(5, 5))

        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, padding='same')
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4, 10))

        self.dropout3 = nn.Dropout(p=0.3)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 2 * 1, 100)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        return x
    
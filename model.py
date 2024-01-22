import torch
from torch import nn
import random
class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        #mu_mix = mu*lmda + mu2 * (1-lmda)
        mu_mix = torch.lerp(mu2, mu, lmda)
        #sig_mix = sig*lmda + sig2 * (1-lmda)
        sig_mix = torch.lerp(sig2, sig, lmda)
        return x_normed*sig_mix + mu_mix

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

class BaselineDCASECNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.activation_1 = nn.ReLU()
        
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.activation_2 = nn.ReLU()
        
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.dropout_1 = nn.Dropout(0.5)
        
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.activation_3 = nn.ReLU()
        
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_2 = nn.Dropout(0.5)
        
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(480, 100)
        self.dropout_3 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(100, 10)

    def forward(self, x):
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
    

    
class BaselineMLPNetwork(nn.Module):
        "No cnn, just a simple MLP"
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(2816, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            x = self.flatten(x)
            x = self.linear_relu_stack(x)
            predictions = self.softmax(x)
            return predictions



import torch
from torch import nn
class AdaResNorm(nn.Module):
    def __init__(self, c, grad=False, id_norm=None, eps=1e-5):
        super(AdaResNorm, self).__init__()
        self.grad = grad
        self.id_norm = id_norm
        self.eps = torch.Tensor(1, c, 1, 1)
        self.eps.data.fill_(eps)

        if self.grad:
            self.rho = nn.Parameter(torch.Tensor(1, c, 1, 1))
            self.rho.data.fill_(0.5)
            self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        else:
            self.rho = torch.Tensor(1, c, 1, 1)
            self.rho.data.fill_(0.5)

    def forward(self, x):
        self.eps = self.eps.to(x.device)
        self.rho = self.rho.to(x.device)

        identity = x
        ifn_mean = x.mean((1, 3), keepdim=True)
        ifn_var = x.var((1, 3), keepdim=True)
        ifn = (x - ifn_mean) / (ifn_var + self.eps).sqrt()

        res_norm = self.rho * identity + (1 - self.rho) * ifn

        if self.grad:
            return self.gamma * res_norm + self.beta
        else:
            return res_norm



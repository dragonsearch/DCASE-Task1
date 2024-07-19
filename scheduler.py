from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LRScheduler
class CosineLRSchedulerWarmup(LRScheduler):
    # This is a wrapper for the CosineAnnealingWarmRestarts scheduler, but the first few epochs are linearly warmed up
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, eta_min=0, last_epoch=-1, warmup_epochs=10):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = self.base_lr
        self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_max=eta_max, T_up=warmup_epochs, gamma=1.0, eta_min=eta_min, last_epoch=last_epoch)
    def get_last_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return self.warmup_lr
        else:
            return self.scheduler.get_last_lr()
    def step(self, epoch=None):
        if epoch < self.warmup_epochs:
            # Increasing linearly from 0, base_lr is 0
            self.warmup_lr 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.warmup_lr
        else:
            self.scheduler.step(epoch)

import torch
from timm.scheduler import CosineLRScheduler

# Test for first 30 epochs and plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    optimizer = torch.optim.SGD([{'params': [torch.nn.Parameter(torch.tensor(1.0))]}], lr=0.01)
    scheduler = CosineLRScheduler(optimizer, t_initial=60, warmup_t=10, warmup_lr_init=0, warmup_prefix=True, cycle_limit=100)
    lrs = []
    for epoch in range(25):
        lr = scheduler._get_lr(epoch)
        print(lr)
        lrs.append(lr)
        lr = scheduler.step(epoch)
    plt.plot(lrs)
    # Save
    plt.savefig("cosine_lr.png")
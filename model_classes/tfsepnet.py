from model_classes.blocks.tfsepconv import TimeFreqSepConvs
import torch.nn as nn
import torch
from model_classes.common import AdaResNorm


defaultcfg = {
    18: ['CONV', 'N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
}

class TfSepNet(torch.nn.Module):
    def __init__(self, depth=18, width=40, dropout_rate=0.2, shuffle=True, shuffle_groups=10):
        super(TfSepNet, self).__init__()
        cfg = defaultcfg[18]
        self.width = width
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.shuffle_groups = shuffle_groups

        self.feature = self.make_layers(cfg)
        i = -1
        while isinstance(cfg[i], str):
            i -= 1
        self.classifier = nn.Conv2d(round(cfg[i] * self.width), 10, 1, bias=True)
        self.alpha = 0.3
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)
        self.eps = 1e-6
        self.p = 0.5

    def freq_mixstyle(self,x):
        if not self.training:
            return x
        if torch.rand(1) > self.p:
            return x
    
        B = x.size(0)

        mu = x.mean(dim=[1, 3], keepdim=True)
        var = x.var(dim=[1, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        # Random domain shuffle
        perm = torch.randperm(B)
        
        mu2, sig2 = mu[perm], sig[perm]
        #mu_mix = mu*lmda + mu2 * (1-lmda)
        mu_mix = torch.lerp(mu2, mu, lmda)
        #sig_mix = sig*lmda + sig2 * (1-lmda)
        sig_mix = torch.lerp(sig2, sig, lmda)
        return x_normed*sig_mix + mu_mix

        

    def make_layers(self, cfg):
        layers = []
        vt = 2
        for v in cfg:
            if v == 'CONV':
                layers += [nn.Conv2d(1, 2 * self.width, 5, stride=2, bias=False, padding=2)]
            elif v == 'N':
                layers += [AdaResNorm(c=round(vt * self.width), grad=True)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [
                    TimeFreqSepConvs(in_channels=round(vt * self.width), out_channels=round(v * self.width),
                                            dropout_rate=self.dropout_rate, shuffle=self.shuffle,
                                            shuffle_groups=self.shuffle_groups)]
                vt = v
            else:
                layers += [
                    TimeFreqSepConvs(in_channels=round(vt * self.width), out_channels=round(vt * self.width),
                                            dropout_rate=self.dropout_rate, shuffle=self.shuffle,
                                            shuffle_groups=self.shuffle_groups)]
        for l in layers:
            self.init_weights(l)

                
        return nn.Sequential(*layers)
    def init_weights(self,l):
        if type(l) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            if l.bias is not None:
                torch.nn.init.constant_(l.bias, 0)
        elif type(l) == TimeFreqSepConvs:
            torch.nn.init.kaiming_normal_(l.freq_dw_conv.conv.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(l.temp_dw_conv.conv.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(l.freq_pw_conv.conv.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(l.temp_pw_conv.conv.weight, mode='fan_out', nonlinearity='relu')
            if l.freq_dw_conv.conv.bias is not None:
                torch.nn.init.constant_(l.freq_dw_conv.conv.bias, 0)
            if l.temp_dw_conv.conv.bias is not None:
                torch.nn.init.constant_(l.temp_dw_conv.conv.bias, 0)
            if l.freq_pw_conv.conv.bias is not None:
                torch.nn.init.constant_(l.freq_pw_conv.conv.bias, 0)
            if l.temp_pw_conv.conv.bias is not None:
                torch.nn.init.constant_(l.temp_pw_conv.conv.bias, 0)

    def forward(self, x):
        x = self.freq_mixstyle(x)
        x = self.feature(x)
        y = self.classifier(x)
        y = y.mean((-1, -2), keepdim=False)
        return y
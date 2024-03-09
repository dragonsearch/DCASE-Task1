import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=False,
                 use_relu=False):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_relu:
            self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.activation(x)
        return x


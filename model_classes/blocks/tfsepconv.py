import torch.nn as nn
import torch
from model_classes.blocks.convblock import ConvBlock
#Import dropblock from torchvision
from torchvision.ops import DropBlock2d
class TimeFreqSepConvs(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.2,
                 shuffle=False,
                 shuffle_groups=10):
        super(TimeFreqSepConvs, self).__init__()
        self.transition = in_channels != out_channels
        self.shuffle = shuffle
        self.half_channels = out_channels // 2

        if self.transition:
            self.trans_conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        use_bn=False, use_relu=False)
        self.freq_dw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels,
                                      kernel_size=(3, 1),
                                      padding=(1, 0), groups=self.half_channels, use_bn=False, use_relu=False)
        self.temp_dw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels,
                                      kernel_size=(1, 3),
                                      padding=(0, 1), groups=self.half_channels, use_bn=False, use_relu=False)
        self.freq_pw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels, kernel_size=1,
                                      use_bn=True, use_relu=True)
        self.temp_pw_conv = ConvBlock(in_channels=self.half_channels, out_channels=self.half_channels, kernel_size=1,
                                      use_bn=True, use_relu=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        # Use dropblock instead of dropout
        self.dropblock = DropBlock2d(block_size=7,p=0.1)
        self.shuffle_layer = ShuffleLayer(group=shuffle_groups)

    def forward(self, x):
        # Expand or shrink channels if in_channels != out_channels
        if self.transition:
            x = self.trans_conv(x)
        # Channel shuffle
        if self.shuffle:
            x = self.shuffle_layer(x)
        # Split feature maps into two halves on the channel dimension
        x1, x2 = torch.split(x, self.half_channels, dim=1)
        # Copy x1, x2 for residual path
        identity1 = x1
        identity2 = x2
        # Frequency-wise convolution block
        #Print statements to debug

        x1 = self.freq_dw_conv(x1)

        x1 = x1.mean(2, keepdim=True)  # frequency average pooling

        x1 = self.freq_pw_conv(x1)

        x1 = self.dropout(x1)
        #x1 = self.dropblock(x1)
        x1 = x1 + identity1
        # Time-wise convolution block
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout(x2)
        #x2 = self.dropblock(x2)
        x2 = x2 + identity2
        # Concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        return x

class ShuffleLayer(nn.Module):
    def __init__(self, group=10):
        super(ShuffleLayer, self).__init__()
        self.group = group

    def forward(self, x):
        b, c, f, t = x.data.size()
        assert c % self.group == 0
        group_channels = c // self.group

        x = x.reshape(b, group_channels, self.group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x
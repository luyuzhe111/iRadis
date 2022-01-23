import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        input_size = (64, 64)
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_tmp = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn_tmp = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.bn_tmp(self.conv_tmp(x)) if self.in_channels != self.out_channels else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + residual)
        return x

class TransposeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        input_size = (64, 64)
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_tmp = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0)
        self.bn_tmp = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.bn_tmp(self.conv_tmp(x)) if self.in_channels != self.out_channels else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + residual)
        return x


class LeakyResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        input_size = (64, 64)
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_tmp = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn_tmp = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        residual = self.bn_tmp(self.conv_tmp(x)) if self.in_channels != self.out_channels else x
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.lr2(self.bn2(self.conv2(x)) + residual)
        return x

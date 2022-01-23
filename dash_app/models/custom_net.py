import torch.nn as nn
import numpy as np
import csv
import torch
import wandb
import time
import collections
import os
import random
from models.resblock import ResBlock
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from PIL import *
from tqdm import tqdm
from math import log2

INPUT_SIZE = (128, 128)
OUT_CHANNELS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
DROP_OUT = 0.5
BETA1 = 0.5
BETA2 = 0.999

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, BLCK1_FILTER, drop_out=0.5):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.BLCK1_FILTER = BLCK1_FILTER
        self.steps = int(log2(INPUT_SIZE[0]))
        filter_in = in_channels
        filter_out = self.BLCK1_FILTER * (2 ** 0)

        self.feature_extract = nn.Conv2d(in_channels, filter_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.high_level_feature_extraction = nn.Sequential(
            nn.BatchNorm2d(filter_out),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((3,3), (2,2), (1, 1)),
        )
       
     
        self.layers = []
        for i in range(self.steps - 1):
            filter_in = filter_out
            filter_out = filter_in * 2
            cur_block = nn.Sequential(
                nn.Conv2d(filter_in, filter_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(filter_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d((3,3), (2,2), (1, 1)),
            )
            self.layers.append(cur_block)
       
        self.layers.append(nn.Flatten())
        self.low_level_feature_extraction = nn.Sequential(*self.layers)

        self.out = nn.Sequential(
            nn.Linear((2 ** int(log2(INPUT_SIZE[0] // 2))) * self.BLCK1_FILTER, self.BLCK1_FILTER*16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(self.BLCK1_FILTER*16, self.BLCK1_FILTER*4),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(self.BLCK1_FILTER*4, self.out_channels)
        )

    def forward(self, x):
        features = self.feature_extract(x)
        features = self.high_level_feature_extraction(features)
        features = self.low_level_feature_extraction(features)
        return self.out(features)
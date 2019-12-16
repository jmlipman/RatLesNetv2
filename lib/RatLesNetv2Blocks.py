from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, Sigmoid
import numpy as np
import torch

class ResNet(nn.Module):
    def __init__(self, in_filters):
        super(ResNet, self).__init__()

        self.seq = nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, in_filters, 3, padding=1),
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, in_filters, 3, padding=1)
            )

    def forward(self, x):
        return x + self.seq(x)

    def __str__(self):
        return "ResNet"

class Bottleneck(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Bottleneck, self).__init__()

        self.seq = nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, out_filters, 1)
            )

    def forward(self, x):
        return self.seq(x)

    def __str__(self):
        return "Bottleneck"



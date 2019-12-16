import torch
from torch import nn
from torch.nn import Conv3d
from lib.RatLesNetv2Blocks import *
from torch.nn.functional import interpolate

class RatLesNetv2(nn.Module):

    def __init__(self, modalities, filters):
        super(RatLesNetv2, self).__init__()

        self.conv1 = Conv3d(modalities, filters, 1)

        self.block1 = ResNet(filters)
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = ResNet(filters)
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block3 = ResNet(filters)
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck1 = Bottleneck(filters, filters)
        self.block4 = ResNet(filters*2)

        self.bottleneck2 = Bottleneck(filters*2, filters)
        self.block5 = ResNet(filters*2)

        self.bottleneck3 = Bottleneck(filters*2, filters)
        self.block6 = ResNet(filters*2)

        self.bottleneck4 = Bottleneck(filters*2, 2)

    def forward(self, x):
        x = self.conv1(x)
        block1_out = self.block1(x)
        block1_size = block1_out.size()

        x = self.mp1(block1_out)
        block2_out = self.block2(x)
        block2_size = block2_out.size()

        x = self.mp2(block2_out)
        block3_out = self.block3(x)
        block3_size = block3_out.size()

        x = self.mp3(block3_out)
        b1 = self.bottleneck1(x)

        x = interpolate(b1, block3_size[2:], mode="trilinear")
        x = torch.cat([x, block3_out], dim=1)

        block4_out = self.block4(x)
        b2 = self.bottleneck2(block4_out)

        x = interpolate(b2, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        block5_out = self.block5(x)
        b3 = self.bottleneck3(block5_out)

        x = interpolate(b3, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        block6_out = self.block6(x)
        b4 = self.bottleneck4(block6_out)

        softed = torch.functional.F.softmax(b4, dim=1)
        # Must be a tuple
        return (softed, )


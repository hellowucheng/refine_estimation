from typing import Any

import torch
import torch.nn as nn
from utils.misc import isExists


class Base(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(Base, self).__init__()
        self.name = 'Base'
        self._exists = isExists

    def load(self, resume=None, pretrained=None):
        if self._exists(resume):
            print('resume training from:', resume)
            self.load_state_dict(torch.load(resume, map_location=lambda storage, loc: storage))
        elif self._exists(pretrained):
            print('load pretrained model:', pretrained)
            self.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage))
        else:
            print('do not have pretrained model')

    def save(self, path):
        torch.save(self.state_dict(), path)


class DUC(Base):

    def __init__(self, in_planes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
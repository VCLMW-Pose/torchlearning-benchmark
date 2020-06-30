# Copyright (c) MW-Pose Group, 2020
import torch.nn as nn
from engine.modeling import registry

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


@registry.BACKBONE.Registry("VGG")
class VGG(nn.Module):

    def __init__(self, in_channel, stages, batch_norm):
        super().__init__()
        layers = list()
        for stage in stages:
            if stage == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            layers.append(nn.Conv2d(in_channel, stage, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(stage))
            layers.append(nn.ReLU(inplace=True))
            in_channel = stage

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

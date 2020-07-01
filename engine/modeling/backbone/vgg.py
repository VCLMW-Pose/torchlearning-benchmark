# Copyright (c) MW-Pose Group, 2020
import torch.nn as nn
from engine.modeling import registry
from engine.modeling.layers import build_norm

# Example stages for VGG13, VGG16, VGG19
# cfg = {
# 'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
# 'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
# 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
# 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# }


@registry.BACKBONE.register("VGG")
class VGG(nn.Module):

    def __init__(self, in_channel, stages, norm, **kwargs):
        super().__init__()
        layers = list()
        for stage in stages:
            if stage == 'M':
                layers.append(nn.MaxPool2d(
                    kernel_size=2,
                    stride=2))
                continue
            elif stage == "a":
                layers.append(nn.AvgPool2d(
                    kernel_size=2,
                    stride=2))
                continue

            layers.append(nn.Conv2d(
                in_channel,
                stage,
                kernel_size=3,
                padding=1))
            if norm:
                layers.append(build_norm(
                    in_channel=stage,
                    norm_layer=norm,
                    **kwargs))
            layers.append(nn.ReLU(inplace=True))
            in_channel = stage

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

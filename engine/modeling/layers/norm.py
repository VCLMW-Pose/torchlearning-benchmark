# Copyright (c) MW-Pose Group, 2020
import torch.nn as nn

norm_factorys = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
    "InstanceNorm": nn.InstanceNorm1d
}


def build_norm(in_channel, norm_layer=None, eps=1e-5, momentum=0.1):
    if norm_layer:

    return None
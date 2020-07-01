# Copyright (c) MW-Pose Group, 2020
import torch.nn as nn

norm_factorys = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
    "InstanceNorm1d": nn.InstanceNorm1d
}


def build_norm(in_channel, norm_layer=None, **kwargs):
    """

    :param in_channel:
    :param norm_layer:
    :param kwargs:
    :return:
    """
    if norm_layer is None:
        return None
    try:
        factory = norm_factorys[norm_layer]
    except Exception:
        raise KeyError("Unknown normalization {}".format(norm_layer))

    if "Batch" in norm_layer:
        pass
    elif "Group" in norm_layer:
        pass
    elif "Layer" in norm_layer:
        pass
    elif "Instance" in norm_layer:
        pass

    return None

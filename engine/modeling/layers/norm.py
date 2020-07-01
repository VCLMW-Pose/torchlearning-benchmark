# Copyright (c) MW-Pose Group, 2020
import torch.nn as nn

norm_factorys = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
}


def build_norm(in_channel, norm_layer=None, **kwargs):
    """
    To build a batch normalization instance, the kwargs configurations are:
    {"eps":1e-5, "momentum":0.1, "affine":True, track_running_stats}
    To build a group normalization instance, the kwargs configurations are:
    {"num_groups_div":10("num_groups":10), "eps":1e-5, "affine":True}
    To build a instance normalization instance, the kwargs configurations are:
    {"eps":1e-5, "momentum":0.1, "affine":False, "track_running_stats":False}

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
        return factory(num_features=in_channel, **kwargs)
    elif "Group" in norm_layer:
        if "num_groups_div" in kwargs:
            _div = kwargs.pop("num_groups_div")
            _num = in_channel // _div
            kwargs["num_groups"] = _num

        return factory(num_channels=in_channel, **kwargs)
    elif "Layer" in norm_layer:
        raise NotImplementedError("Layer normalization is currently not supported.")
    elif "Instance" in norm_layer:
        return factory(num_features=in_channel, **kwargs)

    return None

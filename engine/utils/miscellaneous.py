# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ctypes import Union

import numpy as np

import errno
import torch
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def dict2device(data: Union[dict, list, torch.Tensor], device):
    """Convert torch.Tensor dictionary container to designated device"""
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.to(device)
        return data
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
        return data
    elif isinstance(data, list):
        return [o.to(device) for o in data]

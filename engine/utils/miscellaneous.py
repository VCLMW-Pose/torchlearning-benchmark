# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
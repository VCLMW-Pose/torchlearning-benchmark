# Copyright (c) MW-Pose Group, 2020
import torch


class BaseStructure(object):
    """
    Base target structure class. In torchlearning-benchmark tarining target
    is kept in TargetContainer classes, such as:
        ImageTargetContainer,
        RegistryTargetContainer, etc
    The container branches from Python dictionary, and we store target as
    Structures, like Transforms, Keypoints, BBoxList etc. Any custom target
    structure must inherit BaseStructure and implement all specific input
    data transforms, for example resize, transpose, crop for CV task
    ground truth.

    Only the member 'data' accept valid target data, and it can be any basic
    data type(int, float, double, str, np.ndarray, torch.Tensor). Other types
    are currently not supported by batch collate function, inform me if you
    raise any requirements on torchlearning-benchamrk Structures.
    """
    def __init__(self, data):
        if not isinstance(data, str):
            device = torch.device("cpu") if not hasattr(data, "device") else data.device
            self.data = torch.as_tensor(data, device=device)
        else:
            self.data = data

    def to(self, device):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(device)

        return self


class TargetContainer(dict):
    def __init__(self):
        super(TargetContainer, self).__init__()


class RegistryTargetContainer(dict):
    def __init__(self):
        super(RegistryTargetContainer, self).__init__()


class ImageTargetContainer(dict):
    def __init__(self):
        super(ImageTargetContainer, self).__init__()

    def resize(self, size, *args, **kwargs):
        for k, v in self.items():
            if not isinstance(v, torch.Tensor):
                v.resize(size, *args, **kwargs)

        return self

    def transpose(self, method):
        for k, v in self.items():
            if not isinstance(v, torch.Tensor):
                v.transpose(method)

        return self

    def crop(self, box):
        for k, v in self.items():
            if not isinstance(v, torch.Tensor):
                v.crop(box)

        return self

    def to(self, device):
        for k, v in self.items():
            if hasattr(v, "to"):
                v.to(device)

        return self

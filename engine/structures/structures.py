# Copyright (c) MW-Pose Group, 2020
import torch


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


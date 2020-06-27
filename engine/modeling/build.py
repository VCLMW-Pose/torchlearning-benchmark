# Copyright (c) MW-Pose Group, 2020

from .network import (
    MNISTNet
)

_META_ARCHITECTURES = {"MNISTNet": MNISTNet}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.MODEL.META_ARCH]
    return meta_arch(cfg)

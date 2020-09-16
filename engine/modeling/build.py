# Copyright (c) MW-Pose Group, 2020

from .network import (
    RPMNetEarlyFusion,
    MNISTNet,
    DeSeqNet,
)

_META_ARCHITECTURES = {"MNISTNet": MNISTNet,
                       "DeSeqNet": DeSeqNet,
                       "RPMNetEarlyFusion": RPMNetEarlyFusion,
                       }


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.MODEL.META_ARCH]
    return meta_arch(cfg)

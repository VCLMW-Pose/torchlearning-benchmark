# Copyright (c) 2020 MW-Pose Group, 2020
from engine.modeling import registry


def build_backbone(cfg):
    assert cfg.META_ARCH in registry.BACKBONE, \
        "cfg.CONV_BODY: {} are not registered in registry".format(
            cfg.META_ARCH
        )
    return registry.BACKBONE[cfg.CONV_BODY](cfg)

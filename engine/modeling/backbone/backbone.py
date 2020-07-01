# Copyright (c) 2020 MW-Pose Group, 2020
from engine.modeling import registry

import engine.modeling.backbone.vgg
import engine.modeling.backbone.duc
import engine.modeling.backbone.mlp


def build_backbone(backbone, **kwargs):
    assert backbone in registry.BACKBONE, \
        "cfg.backbone: {} are not registered in registry".format(
            backbone
        )
    return registry.BACKBONE[backbone](**kwargs)

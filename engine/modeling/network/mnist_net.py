# Copyright (c) MW-Pose Group, 2020

import json
import torch
import torch.nn as nn

from engine.modeling.backbone import build_backbone


class MNISTNet(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg (yacs.CfgNode): Session configuration
        """
        super(MNISTNet, self).__init__()
        with open(cfg.MODEL.CONFIG, 'r') as f:
            self.config = json.load(f)

        backbone = self.config["backbone"]
        opt = self.config["options"]
        args = dict(
            input_dims=opt["input"],
            n_hiddens=opt["mlp_hiddens"],
            n_class=opt["classes"]
        )

        self.mlp = build_backbone(backbone, **args)

    def forward(self, input):
        return self.mlp(input)

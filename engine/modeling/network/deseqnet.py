# Copyright (c) MW-Pose Group, 2020
from engine.modeling.backbone import build_backbone

import torch.nn as nn
import json


class DeSeqNet(nn.Module):
    def __init__(self, cfg):
        """

        :param cfg:
        """
        super(DeSeqNet, cfg).__init__()
        with open(cfg.MODEL.CONFIG, 'r') as f:
            self.config = json.load(f)
            f.close()
        encoder = self.config["encoder"]
        linear = self.config["linear"]
        decoder = self.config["decoder"]
        self.hms_size = self.config["hms_size"]

        self.encoder = build_backbone(
            backbone=encoder["backbone"],
            in_channel=encoder["in_channel"],
            stages=encoder["stages"],
        )
        self.linear = build_backbone(
            backbone=linear["backbone"],
            in_channel=linear["in_channel"],
            n_hiddens=linear["n_hiddens"],
            out_channel=linear["out_channel"],
            p_dropout=linear["p_dropout"],
        )
        self.decoder = build_backbone(
            backbone=decoder["backbone"],

        )
        self.out = nn.Conv2d(
            in_channels=decoder["stages"][-1],
            out_channels=self.config["num_keypoints"],
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, image, target):
        image = self.encoder

        if target:
            pass


# Copyright (c) MW-Pose Group, 2020

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            f.close()

        backbone = self.config["backbone"]
        opt = self.config["options"]
        args = dict(
            input_dims=opt["input"],
            n_hiddens=opt["mlp_hiddens"],
            n_class=opt["classes"]
        )

        self.mlp = build_backbone(backbone, **args)

    def forward(self, image, target=None):
        """
        Args:
            image (torch.Tensor): Mini-batch images
            target (ImageTargetContainer): collated target
        """
        logistic = self.mlp(image)
        # Compute loss
        if target:
            labels = target["labels"]
            loss = F.cross_entropy(logistic, labels)
            return loss

        # Get classification prediction
        pred = logistic.data.max(1).indices
        return pred



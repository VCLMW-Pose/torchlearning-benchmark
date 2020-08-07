# Copyright (c) MW-Pose Group, 2020
from engine.modeling.backbone import build_backbone
from engine.structures.heatmap import heatmap_to_coord_simple

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import json


class DeSeqNet(nn.Module):
    def __init__(self, cfg):
        """

        :param cfg:
        """
        super(DeSeqNet, self).__init__()
        with open(cfg.MODEL.CONFIG, 'r') as f:
            self.config = json.load(f)
            f.close()
        encoder = self.config["encoder"]
        linear = self.config["linear"]
        decoder = self.config["decoder"]
        self.hms_size = np.array(self.config["hms_size"])

        args = {
            "backbone": encoder["backbone"],
            "in_channels": encoder["in_channel"],
            "stages": encoder["stages"],
            "norm": encoder["norm"],
            **encoder["norm_options"],
        }
        self.encoder = build_backbone(**args)
        self.linear = build_backbone(
            backbone=linear["backbone"],
            in_channel=linear["in_channel"],
            n_hiddens=linear["n_hiddens"],
            out_channel=linear["out_channel"],
            p_dropout=linear["p_dropout"],
        )
        args = {
            "backbone": decoder["backbone"],
            "in_channels": decoder["in_channel"],
            "stages": decoder["stages"],
            "upscale_factor": decoder["upscale_factor"],
            "norm": decoder["norm"],
            **decoder["norm_options"],
        }
        self.decoder = build_backbone(**args)
        self.hms_size = (self.hms_size / (2 ** len(decoder["stages"]))).astype(np.int)

        self.out = nn.Conv2d(
            in_channels=decoder["stages"][-1] // (decoder["upscale_factor"] ** 2),
            out_channels=self.config["num_keypoints"],
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, image, target=None):
        image = self.encoder(image)
        feat_vector = image.view(image.shape[0], -1)
        feat_vector = self.linear(feat_vector)

        hms = feat_vector.view((feat_vector.shape[0], -1, self.hms_size[1], self.hms_size[0]))
        hms = self.decoder(hms)
        hms = self.out(hms)

        if target is not None:
            losses = dict()
            hms_gt = target["hms"]
            hms_mask = target["hms_mask"]

            losses["hms_loss"] = 0.5 * F.mse_loss(hms.mul(hms_mask), hms_gt.mul(hms_mask))
            return losses

        hms = hms.to(torch.device("cpu")).numpy()
        keypoints = list()
        for i in range(hms.shape[0]):
            pose_coords, pose_scores = heatmap_to_coord_simple(
                hms=hms[i],
                bbox=(0, 0) + hms[i].shape[1:3])
            keypoints += [np.concatenate((pose_coords, pose_scores), axis=1)]
        return torch.as_tensor(np.array(keypoints))


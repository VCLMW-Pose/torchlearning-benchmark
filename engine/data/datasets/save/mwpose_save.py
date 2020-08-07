# Copyright (c) MW-Pose Group, 2020
from engine.structures.keypoints import MWPoseKeypoints
import numpy as np
import logging
import torch
import cv2
import os


def do_mwpose_visualization(dataset, predictions, output_folder):
    # TODO: Compute MSCOCO OKS metrics
    # TODO: Compute MPii PCKh metrics
    # thre = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])

    for idx, (k, dt) in enumerate(predictions.items()):
        dt = np.array(dt)

        n = dt.shape[0]
        for i in range(n):
            keypoint = MWPoseKeypoints(dt[i, ::], size=(72, 128))

            img = keypoint.drawpose()
            cv2.imwrite(os.path.join(output_folder, "%d.jpg" % (n * idx + i)), img)


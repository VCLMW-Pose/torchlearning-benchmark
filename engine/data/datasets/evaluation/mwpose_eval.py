# Copyright (c) MW-Pose Group, 2020
import numpy as np
import logging
import torch


def do_mwpose_evaluation(dataset, predictions, gts, output_folder):
    # TODO: Compute MSCOCO OKS metrics
    # TODO: Compute MPii PCKh metrics
    logger = logging.getLogger("torchlearning-benchmark")
    names = dataset.names
    thre = np.array([0.05, 0.10, 0.20, 0.50, 0.75, 1.00, 2.50, 5.00])
    pck = np.zeros((len(gts), len(names), len(thre)))
    total = 0
    # thre = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])

    for idx, (k, dt) in enumerate(predictions.items()):
        gt = gts[k]
        gt = gt["keypoints"]
        dt = np.array(dt)
        gt = np.array(gt)

        hxg = gt[:, 0, 0] - gt[:, 1, 0]
        hyg = gt[:, 0, 1] - gt[:, 1, 1]
        h = 2 * np.sqrt(hxg ** 2 + hyg ** 2)

        n = dt.shape[0]
        m = dt.shape[1]
        e = np.zeros(m)
        for i in range(n):
            xg = gt[i, :, 0]
            yg = gt[i, :, 1]
            xd = dt[i, :, 0]
            yd = dt[i, :, 1]
            dx = xd - xg
            dy = yd - yg

            e[:] = np.sqrt(dx ** 2 + dy ** 2) / h[i]

            for j in range(len(thre)):
                pck[idx, :, j] += e[:] <= thre[j]

        total += n

    pck = np.sum(pck, axis=0) / total
    for idx, name in enumerate(names):
        log_str = ""
        for j, th in enumerate(thre):
            log_str += " PCKh@{}_{}: {} ".format(name, th, pck[idx, j])

        logger.info(log_str)

    return pck

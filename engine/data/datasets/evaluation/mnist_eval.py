# Copyright (c) MW-Pose Group, 2020

import logging
import torch


def do_mnist_evaluation(predictions, gts, output_folder):
    """
    Args:
        predictions (dict): Predictions on test set, the key is batch id
        gts (dict): Ground truth for every batch
    """
    logger = logging.getLogger("torchlearning-benchmark")
    accuracy = 0
    denominator = 0

    for k in predictions.keys():
        pred = predictions[k]
        gt = gts[k]

        accuracy += torch.sum(pred == gt).numpy()
        denominator += pred.shape[0]

    accuracy /= float(denominator)
    logger.info(
        "Evaluation on MNIST test set, accuracy: {}".format(accuracy)
    )

    return accuracy

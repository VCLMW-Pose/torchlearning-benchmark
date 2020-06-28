# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from engine.data.datasets.evaluation import evaluate
from engine.utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = dict()
    gts = dict()
    cpu_device = torch.device("cpu")
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets = batch
        with torch.no_grad():
            if timer:
                timer.tic()
                output = model(images.to(device))
            if timer:
                timer.toc()

            if isinstance(output, dict):
                for k, v in output.items():
                    output[k] = v.to(cpu_device)
            elif isinstance(output, list):
                output = [o.to(cpu_device) for o in output]
            else:
                output = output.to(cpu_device)
        results_dict.update(
            {idx: output}
        )
        gts.update(
            {idx: targets}
        )
    return results_dict, gts


def inference(
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    logger = logging.getLogger("torchlearning-benchmark")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions, gts = compute_on_dataset(model, data_loader, device, inference_timer)

    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {}".format(
            total_time_str,
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ".format(
            total_infer_time,
        )
    )

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    gts=gts,
                    output_folder=output_folder)

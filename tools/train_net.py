# Copyright (c) MW-Pose Group, 2020
# Modified from maskrcnn-benchmark train_net.py


import argparse
import os

import torch

from engine.config import cfg
from engine.trainer import do_train
from engine.inference import inference
from engine.modeling import build_model
from engine.data import build_data_loader
from engine.utils.logger import setup_logger
from engine.utils.checkpoint import Checkpointer
from engine.solver import build_optimizer, build_lr_scheduler


def train(cfg):
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = cfg.SAVE_TO_DISK

    checkpointer = Checkpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = build_data_loader(
        cfg,
        is_train=True,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = build_data_loader(cfg, is_train=False, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model


def run_test(cfg, model):
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            os.mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = build_data_loader(cfg, is_train=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()


    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.mkdir(output_dir)

    logger = setup_logger("torchlearning_benchmark", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # save overloaded model config in the output directory
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    with open(output_config_path, 'w') as f:
        f.write(cfg.dump())

    model = train(cfg)

    if not args.skip_test:
        run_test(cfg, model)


if __name__ == "__main__":
    main()

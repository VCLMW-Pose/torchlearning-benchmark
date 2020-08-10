# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy


class DatasetCatalog(object):
    DATA_DIR = "../datasets"
    DATASETS = {
        "mnist_train": {
            "img_dir": "mnist/train",
            "ann_file": "mnist/train/labels"
        },
        "mnist_test": {
            "img_dir": "mnist/test",
            "ann_file": "mnist/test/labels"
        },
        "mwpose_train": {
            "root": "mwpose/",
            "size": (360, 640),
            "mode": "mwpose"
        },
        "mwpose_test": {
            "root": "mwpose/",
            "size": (360, 640),
            "mode": "mwpose"
        },
        "mwpose_throughwall": {
            "root": "mwpose_throughwall/",
            "size": (360, 640),
            "mode": "mwpose"
        }
    }

    @staticmethod
    def get(name):
        if "mnist" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="MNIST",
                args=args,
            )
        elif "mwpose" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
                size=attrs["size"],
                mode=attrs["mode"]
            )
            return dict(
                factory="MWPose",
                args=args,
            )
        # elif "your dataset name" in name:
            # data_dir = DatasetCatalog.DATA_DIR
            # attrs = DatasetCatalog.DATASETS[name]
            # args = dict(
            #     data_dir=os.path.join(data_dir, attrs["data_dir"]),
            #     split=attrs["split"],
            # )
            # return dict(
            #     factory="your data set implementation",
            #     args=args,
            # )
        raise RuntimeError("Dataset not available: {}".format(name))

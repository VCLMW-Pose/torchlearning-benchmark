# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "mnist_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "mnist_test": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        }
    }

    @staticmethod
    def get(name):
        if "mnist" in  name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_files"]),
            )
            return dict(
                factory="MNIST",
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

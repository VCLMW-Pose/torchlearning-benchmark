# Copyright (c) 2020 MW-Pose Group, 2020
from torch.utils.data import Dataset

import engine.structures.keypoints as K

import numpy as np
import json
import os


class MWPose(Dataset):
    """MWPose data set"""
    def __init__(self, root, size, transforms=None, mode='mwpose', is_train=False):
        """
        Args:
            ann_file (string): Annotation directory.
            root (string): Root directory of data set.
            transforms (callable, optional): Transforms should be applied to
                sampled mini-batch.
            is_train (bool): Whether is train set.
        """
        super(MWPose, self).__init__()

        self.root = root
        self.size = size
        self.is_train = is_train

        # sig_dir: root/signals/  ann_dir: root/labels/
        self.sig_dir = os.path.join(root, "signals")
        self.ann_dir = os.path.join(root, "labels")

        txt = "train.txt" if is_train else "valid.txt"
        try:
            with open(os.path.join(self.root, txt), 'r') as f:
                self.sig_names = f.readlines()
        except IOError:
            raise IOError("Cannot open data set list from {}"
                          .format(os.path.join(self.root, txt)))

        # Yield the paths of RF images and annotations
        self.sig_paths = [os.path.join(root, "signals/" + name) for name in self.sig_names]
        self.ann_paths = [os.path.join(root, "labels/" + name + ".txt") for name in self.sig_names]

        # Read keypoints names
        with open(os.path.join(self.root, "keypoint.names"), 'r') as f:
            self.names = f.readlines()

        # Key point factory
        if mode == 'mwpose':
            self._factory = getattr(K, "MWPoseKeypoints")
        elif mode == "COCO":
            self._factory = getattr(K, "PersonKeypoints")
        else:
            raise NotImplementedError("Key point mode not implemented.")

        # Transforms
        self._transforms = transforms

    def __getitem__(self, idx):
        rf_image = self._read_rf_image(self.sig_paths[idx])
        anno = open(self.ann_paths[idx], 'r')
        anno = json.load(anno)

        assert len(anno) == 1, 'Multi-person scenario not implemented.'
        anno = anno[0][self.names]

        # Key point ground truth
        keypoints = np.array(anno).astype(np.float)
        keypoints[..., 2] = 1

        keypoints = self._factory(keypoints, self.size)

    def __len__(self):
        return len(self.sig_paths)

    @staticmethod
    def _read_rf_image(dir):
        # Open the directory in the form of read only binary file
        sig_file = open(dir, mode='rb')
        data = np.fromfile(sig_file, dtype=np.int32)

        # Dimensions
        try:
            _x, _y, _z = data[0:2]
        except Exception:
            raise IOError("RF Image does not exist, excepetion occurred at directory: {}"
                  .format(dir))

        # Resize the signal as size_z x size_x x size_y
        rf_image = np.array(data[3:]).reshape((_x, _y, _z))

        return rf_image

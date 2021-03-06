# Copyright (c) 2020 MW-Pose Group, 2020
from torch.utils.data import Dataset

from engine.structures.structures import ImageTargetContainer as Container
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
        self.sig_paths = [os.path.join(root, "signals/" + name.strip()) for name in self.sig_names]
        self.ann_paths = [os.path.join(root, "labels/" + name.strip() + ".json") for name in self.sig_names]

        # Read keypoints names
        with open(os.path.join(self.root, "keypoint.names"), 'r') as f:
            self.names = f.readlines()
            self.names = [name.strip() for name in self.names]

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
        target = Container()

        assert len(anno) == 1, 'Multi-person scenario not implemented. %s' % self.ann_paths[idx]
        anno = anno[0]
        anno = [anno[name][:] for name in self.names]

        # Key point ground truth
        keypoints = np.array(anno).astype(np.float)
        keypoints[..., 2] = 1
        keypoints = self._factory(keypoints, self.size)

        # Heatmap ground truth for training, key points for evaluation
        # if self.is_train:
        #     heatmap = keypoints.gen_heatmap()
        #     target["heatmap"] = heatmap
        # else:
        #     target["keypoints"] = keypoints
        target["keypoints"] = keypoints

        if self._transforms:
            return self._transforms(rf_image, target)

    def __len__(self):
        return len(self.sig_paths)

    @staticmethod
    def _read_rf_image(dir):
        # Open the directory in the form of read only binary file
        sig_file = open(dir, mode='rb')
        data = np.fromfile(sig_file, dtype=np.int32)

        # Dimensions
        try:
            _x, _y, _z = data[0:3]
        except Exception:
            raise IOError("RF Image does not exist, excepetion occurred at directory: {}"
                  .format(dir))

        # Resize the signal as size_z x size_x x size_y
        rf_image = np.array(data[3:]).reshape((_x, _y, _z))
        rf_image = np.fliplr(rf_image).astype(np.float32) / 255.
        rf_image = rf_image.copy()

        return rf_image

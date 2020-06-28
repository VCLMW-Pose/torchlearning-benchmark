# Copyright (c) MW-Pose Group, 2020

from torch.utils.data import Dataset

from engine.structures import ImageTargetContainer as Container

import PIL.Image as Image
import numpy as np
import torch
import os


class MNIST(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        """
        Args:
            root (str): Data set root directory
            ann_file (str): Annootation directory, its a folder here
            transforms (T, optional): Composed transforms

        How to create your custom data set:
        """
        super(MNIST, self).__init__()
        self.root = root
        self.ann_file = ann_file

        assert os.path.exists(self.root), \
            'Data set root directory not exists: {}'.format(self.root)
        assert os.path.exists(self.ann_file), \
            'Annotation file(s) not exists: {}'.format(self.ann_file)

        # Read image paths
        self.img_names = os.listdir(os.path.join(self.root, "images"))
        self.img_names = sorted(self.img_names)
        self.img_paths = [os.path.join(self.root, "images/" + name) for name in self.img_names]

        # Read annotation paths
        self.ann_names = os.listdir(self.ann_file)
        self.ann_names = sorted(self.ann_names)
        self.ann_paths = [os.path.join(self.ann_file, name) for name in self.ann_names]

        assert len(self.img_paths) == len(self.ann_paths), \
            'Number of images and annotations is not match'

        self._transform = transforms

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        target = Container()
        with open(self.ann_paths[idx], 'r') as f:
            label = int(f.readline())
            target["labels"] = torch.as_tensor(label)
            f.close()

        if self._transform is not None:
            return self._transform(image, target)

    def __len__(self):
        return len(self.img_names)

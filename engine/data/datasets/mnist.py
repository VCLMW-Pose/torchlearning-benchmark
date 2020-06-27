# Copyright (c) MW-Pose Group, 2020

from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        super(MNIST, self).__init__()
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

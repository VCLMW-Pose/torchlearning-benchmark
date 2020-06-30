# Copyright (c) MW-Pose Group, 2020
from engine.structures.keypoints import Keypoints
from engine.structures.heatmap import HeatmapMask
from engine.structures.heatmap import Heatmap


class RandomHorizontalFlip3D(object):
    def __init__(self, prob):
        self._prob = prob

    def __call__(self, image, target):
        pass


class RandomVerticalFlip3D(object):
    def __init__(self, prob):
        self._prob = prob

    def __call__(self, image, target):
        pass


class CalibrateMWPose(object):
    def __init__(self, trans, rot):
        pass

    def __call__(self, image, target):
        pass


class GenerateHMS(object):
    def __init__(self, hms_size, sigma, pop_kp=False):
        self._hms_size = hms_size
        self._sigma = sigma
        self.pop_kp = pop_kp

    def __call__(self, image, target):
        assert isinstance(target["keypoints"], Keypoints), \
            'Expect Keypoints in target["keypoint"], but get {}'.format(type(target["keypoints"]))

        hms, keypoints_weight = target["keypoints"].gen_heatmap(self._hms_size, self._sigma)
        # target["hms"] = Heatmap(hms)
        # target["hms_mask"] = HeatmapMask(keypoints_weight)
        target["hms"] = hms
        target["hms_mask"] = keypoints_weight

        if self.pop_kp:
            target.pop["keypoints"]

        return image, target


class Resize3D(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, image, target):
        pass



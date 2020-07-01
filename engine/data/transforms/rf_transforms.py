# Copyright (c) MW-Pose Group, 2020
from engine.structures.keypoints import Keypoints
from engine.structures.heatmap import HeatmapMask
from engine.structures.heatmap import Heatmap

import torchvision.transforms.functional
import random
import torch


def _is_tensor_a_torch_image(input):
    return input.ndim >= 2


def vflip(img):
    # type: (Tensor) -> Tensor
    """Vertically flip the given the Image Tensor.
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-2)


def hflip(img):
    # type: (Tensor) -> Tensor
    """Horizontally flip the given the Image Tensor.
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-1)


class RandomHorizontalFlip3D(object):
    def __init__(self, prob):
        self._prob = prob

    def __call__(self, image, target):
        if random.random() < self._prob:
            image = hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip3D(object):
    def __init__(self, prob):
        self._prob = prob

    def __call__(self, image, target):
        if random.random() < self._prob:
            image = vflip(image)
            target = target.transpose(1)
        return image, target


class CalibrateMWPose(object):
    def __init__(self, trans, rot):
        pass

    def __call__(self, image, target):
        pass


class KeypointsResize(object):
    def __init__(self, size):
        self._size = size

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


def pad(img, padding, fill, padding_mode = "constant"):
    r"""Pad the given Tensor Image on all sides with specified padding mode and fill value.
    Args:
        img (Tensor): Image to be padded.
        padding (int or tuple or list): Padding on each border. If a single int is provided this
            is used to pad all borders. If a tuple or list of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple or list of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively. In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        fill (int): Pixel fill value for constant fill. Default is 0.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Only "constant" is supported for Tensors as of now.
            - constant: pads with a constant value, this value is specified with fill
    Returns:
        Tensor: Padded image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", ]:
        raise ValueError("Only constant padding_mode supported for torch tensors")

    if isinstance(padding, int):
        if torch.jit.is_scripting():
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    img = torch.nn.functional.pad(img, p, mode=padding_mode, value=float(fill))
    return img


class Pad3D(object):
    def __init__(self, pad):
        self._pad = pad

    def __call__(self, image, target):
        image = pad(image, self._pad, fill=0, padding_mode="constant")
        return image, target


class Resize3D(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, image, target):
        pass

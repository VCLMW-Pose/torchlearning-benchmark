# Copyright (c) MW-Pose Group, 2020
from . import transforms as T


def build_transforms(cfg, is_train=True):
    """
    Args:
        cfg (yacs CfgNode): Configuration
        is_train (bool): Train set transforms or test set transforms

    How to add new transform to torchlearning-benchmark:

    """
    transform = list()
    trans_configs = cfg.INPUT.TRANSFORMS

    for trans in trans_configs:
        try:
            factory = getattr(T, trans)
        except Exception:
            raise NotImplementedError("Transform {} is not implemented.".format(trans))

        if trans == "Resize":
            min_size = cfg.INPUT.MIN_SIZE_TRAIN if is_train else cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
            transform += [factory(
                min_size=min_size,
                max_size=max_size
            )]
        elif trans == "Normalize":
            transform += [
                factory(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD,
                    to_bgr255=cfg.INPUT.TO_RGB255
                )
            ]
        elif trans == "ColorJitter":
            brightness = cfg.INPUT.BRIGHTNESS if is_train else 0.0
            contrast = cfg.INPUT.CONTRAST if is_train else 0.0
            saturation = cfg.INPUT.SATURATION if is_train else 0.0
            hue = cfg.INPUT.HUE if is_train else 0.0

            transform += [
                factory(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ]
        elif trans == "RandomHorizontalFlip":
            flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN if is_train else 0.0
            transform += [
                factory(
                    prob=flip_horizontal_prob,
                )
            ]
        elif trans == "RandomVerticalFlip":
            flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN if is_train else 0.0
            transform += [
                factory(
                    prob=flip_vertical_prob,
                )
            ]
        elif trans == "ToTensor":
            transform += [
                factory()
            ]

    return T.Compose(transform)

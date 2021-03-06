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
    trans_configs = cfg.INPUT.TRAIN_TRANSFORMS if is_train else cfg.INPUT.TEST_TRANSFORMS

    for trans in trans_configs:
        try:
            factory = getattr(T, trans)
        except Exception:
            raise NotImplementedError("Transform {} is not implemented.".format(trans))

        if trans == "Resize":
            min_size = cfg.INPUT.MIN_SIZE_TRAIN if is_train else cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
            transform += [
                factory(
                    min_size=min_size,
                    max_size=max_size
                )
            ]
        elif trans == "Normalize":
            transform += [
                factory(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD,
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
        elif trans == "RandomHorizontalFlip3D":
            flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN if is_train else 0.0
            transform += [
                factory(
                    prob=flip_horizontal_prob,
                )
            ]
        elif trans == "RandomVerticalFlip3D":
            flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN if is_train else 0.0
            transform += [
                factory(
                    prob=flip_vertical_prob,
                )
            ]
        elif trans == "Pad3D":
            pad = cfg.INPUT.PAD_TRAIN if is_train else cfg.INPUT.PAD_TEST
            transform += [
                factory(
                    pad=pad,
                )
            ]
        elif trans == "GenerateHMS":
            # if is_train:
            transform += [
                factory(
                    hms_size=cfg.INPUT.HMS_SIZE,
                    sigma=cfg.INPUT.HMS_SIGMA,
                )
            ]
        elif trans == "SplitSourceRef":
            transform += [
                factory()
            ]
        elif trans == "Resampler":
            transform += [
                factory(
                    num=cfg.INPUT.RESAMPLE_NUM,
                )
            ]
        elif trans == "FixedResampler":
            transform += [
                factory(
                    num=cfg.INPUT.RESAMPLE_NUM,
                )
            ]
        elif trans == "RandomJitter":
            transform += [
                factory(
                    scale=cfg.INPUT.PCJITTER_SCALE,
                    clip=cfg.INPUT.PCJITTER_CLIP
                )
            ]
        elif trans == "RandomCrop":
            transform += [
                factory(
                    p_keep=cfg.INPUT.PCCROP_P_KEEP,
                )
            ]
        elif trans == "RandomTransformSE3":
            transform += [
                factory(
                    rot_mag=cfg.INPUT.ROT_MAG,
                    trans_mag=cfg.INPUT.TRANS_MAG,
                    random_mag=cfg.INPUT.RANDOM_MAG
                )
            ]
        elif trans == "RandomTransformSE3_euler":
            transform += [
                factory(
                    rot_mag=cfg.INPUT.ROT_MAG,
                    trans_mag=cfg.INPUT.TRANS_MAG,
                    random_mag=cfg.INPUT.RANDOM_MAG
                )
            ]
        elif trans == "RandomRotatorZ":
            transform += [
                factory()
            ]
        elif trans == "ShufflePoints":
            transform += [
                factory()
            ]
        elif trans == "SetDeterministic":
            transform += [
                factory()
            ]

    return T.Compose(transform)

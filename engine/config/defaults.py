# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCH = "MNISTNet"
_C.MODEL.CONFIG = "../configs/mnist_net/mnist_net.json"

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Add expected transforms to the tuple, and edit transform configurations with
# following settings.
_C.INPUT.TRAIN_TRANSFORMS = ()
_C.INPUT.TEST_TRANSFORMS = ()


# ===========================  Image transforms  ==============================
# --------------------------------  Resize  -----------------------------------
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333

# ------------------------------  Normalize  ----------------------------------
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to RGB format, in range 0-255
_C.INPUT.TO_RGB255 = True

# -----------------------------  ColorJitter  ---------------------------------
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0


# --------------------------  RandomHorizontalFlip  ---------------------------
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5

# ---------------------------  RandomVerticalFlip  ----------------------------
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------  Pad  -----------------------------------
_C.INPUT.PAD_TRAIN = ()
_C.INPUT.PAD_TEST = ()

# -------------------------------  GenerateHMS  -------------------------------
_C.INPUT.HMS_SIZE = ()
_C.INPUT.HMS_SIGMA = 2
_C.INPUT.HMS_POP_KP = True

# ------------------------------- SplitSourceRef ------------------------------
# -------------------------- Resampler/FixedResampler -------------------------
_C.INPUT.RESAMPLE_NUM = 1024

# -------------------------- RandomJitter (Point cloud) -----------------------
_C.INPUT.PCJITTER_SCALE = 0.01
_C.INPUT.PCJITTER_CLIP = 0.05

# --------------------------- RandomCrop (Point cloud) ------------------------
_C.INPUT.PCCROP_P_KEEP = [0.7, 0.7]

# ------------------ RandomTransformSE3/RandomTransformSE3_euler --------------
_C.INPUT.ROT_MAG = 45.0
_C.INPUT.TRANS_MAG = 0.5
_C.INPUT.RANDOM_MAG = False

# ------------------------------- RandomRotatorZ ------------------------------

# ---------------------------------

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SHUFFLE = True
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.OPTIMIZER = "Momentum"
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

# ------------------------------------ Adam ------------------------------------
_C.SOLVER.BETAS = (0.9, 0.999)

_C.SOLVER.SCHEDULER = "WarmupMultiStep"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.TEST_PERIOD = 0

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.SAVE_TO_DISK = True

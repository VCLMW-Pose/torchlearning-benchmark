from engine.modeling.build import build_model
from engine.utils.miscellaneous import mkdir
from engine.config import cfg
from tqdm import tqdm

import argparse
import os


def estimate(cfg, root):
    filenames = os.listdir(root)
    model = build_model(cfg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deseqnet Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/deseqnet_vgg16_mlp_duc3x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--root",
        default="../datasets/mwpose/"
    )
    parser.add_argument(
        "--output_dir",
        default="../models/mwpose_vgg_16_8.7/results"
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = args.output_dir
    if output_dir:
        mkdir(output_dir)

    estimate(cfg, )


from engine.config import cfg
from engine.modeling import build_model


def test_mnistnet(cfg):
    mnistnet = build_model(cfg)
    print(mnistnet)


if __name__ == "__main__":
    cfg.merge_from_file("../configs/simple_alexnet_mnist.yaml")
    test_mnistnet(cfg)
# Copyright (c) MW-Pose Group, 2020
from engine.structures.structures import BaseStructure
import torch.utils.data._utils.collate as collate


class StructureCollator(object):
    """
    Default batch collator used in torchlearning-benchmark,
    designed for TargetContainer and target Structures.
    """
    def __init__(self):
        self.collate_fn = collate.default_collate

    def __call__(self, batch):
        # batch: list[tuple(image:torch.Tensor, Container)]
        for _, container in batch:
            for k, v in container.items():
                if not isinstance(v, BaseStructure):
                    continue
                container[k] = v.data

        return self.collate_fn(batch)

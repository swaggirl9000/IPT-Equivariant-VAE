"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import nn

from src.layers.directions import generate_2d_directions, generate_uniform_directions
from src.layers.ect import EctConfig, compute_ect_point_cloud


@dataclass
class TransformConfig(EctConfig):
    structured: bool


class EctTransform(nn.Module):
    def __init__(self, config: TransformConfig):
        self.config = config

        if not hasattr(config, "structured"):
            structured = False
        else:
            structured = config.structured

        if structured and config.ambient_dimension == 2:
            self.v = generate_2d_directions(config.num_thetas)
        else:
            self.v = generate_uniform_directions(
                config.num_thetas, d=config.ambient_dimension, seed=config.seed
            )
        self.ect_fn = torch.compile(
            partial(
                compute_ect_point_cloud,
                v=self.v,
                radius=self.config.r,
                resolution=self.config.resolution,
                scale=self.config.scale,
            )
        )

    def __call__(self, x):
        return self.ect_fn(x)


class Ect2DTransform:
    def __init__(self, config: EctConfig, device="cpu"):
        self.config = config
        self.v = generate_2d_directions(config.num_thetas).to(device)
        self.ect_fn = torch.compile(
            partial(
                compute_ect_point_cloud,
                v=self.v,
                radius=self.config.r,
                resolution=self.config.resolution,
                scale=self.config.scale,
            )
        )

    def __call__(self, x):
        return self.ect_fn(x)

TRANSFORMDICT = {
    "ECT": EctTransform,
}


def get_transform(config):
    return TRANSFORMDICT[config.name](config)

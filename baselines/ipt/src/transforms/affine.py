"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pydantic
import torch
from torch import nn
from torchvision.transforms import ToTensor

from layers.directions import generate_2d_directions, generate_uniform_directions
from layers.ect import EctConfig, compute_ect_point_cloud


class TransformConfig(pydantic.BaseModel):
    module: str
    mean_mean: tuple[float, float, float]
    mean_std: tuple[float, float, float]
    scale_mean: float
    scale_std: float
    max_scale: float


class Transform(nn.Module):
    """
    Affine transform. Translates point clouds and applies scaling.
    """

    def __init__(self, config: TransformConfig):
        super().__init__()
        self.config = config

        self.mean_mean = nn.Parameter(
            torch.tensor(config.mean_mean), requires_grad=False
        )
        self.mean_std = nn.Parameter(torch.tensor(config.mean_std), requires_grad=False)

        self.scale_mean = nn.Parameter(
            torch.tensor(config.scale_mean), requires_grad=False
        )
        self.scale_std = nn.Parameter(
            torch.tensor(config.scale_std), requires_grad=False
        )

    def __call__(self, x):
        x_mean = x.mean(axis=1, keepdim=True)
        x_max_norm = (x - x_mean).norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]
        x_normalized = (x - x_mean) / x_max_norm
        # New mean
        m = self.mean_std.unsqueeze(0).unsqueeze(0) * torch.randn_like(
            x_mean, device=x.device
        ) + self.mean_mean.unsqueeze(0).unsqueeze(0)
        m_radii = m.norm(dim=-1, keepdim=True)
        m_radii[m_radii > 0.8 * self.config.max_scale] *= 1 / 0.8
        m_radii[m_radii < 0.8 * self.config.max_scale] = 1
        m = m / m_radii

        s = (
            self.scale_std * torch.randn_like(x_max_norm, device=x.device)
            + self.scale_mean
        )

        s_max = self.config.max_scale - m.norm(dim=-1, keepdim=True)
        s_new = torch.clamp(s, min=torch.tensor(0.1, device=x.device), max=s_max)

        x_new = s_new * x_normalized + m

        return x_new

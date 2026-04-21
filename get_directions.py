import torch
import math


def get_directions(num_points: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(0, num_points, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))   

    y = 1.0 - (indices / (num_points - 1)) * 2.0
    radius = torch.sqrt(1.0 - y * y)
    theta = phi * indices

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    dirs = torch.stack([x, y, z], dim=1)    
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    weights = torch.full((num_points,), 1.0 / num_points)

    return dirs, weights

# import quadpy
# import torch

# def get_directions(order: int = 21) -> tuple[torch.Tensor, torch.Tensor]:
#     scheme = getattr(quadpy.u3, f"lebedev_{order:03d}")()
#     dirs    = torch.tensor(scheme.points.T, dtype=torch.float32)   # (N, 3)
#     weights = torch.tensor(scheme.weights,  dtype=torch.float32)   # (N,)
#     weights = weights / weights.sum()
#     return dirs, weights
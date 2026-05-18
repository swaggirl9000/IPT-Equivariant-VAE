import torch
from torch import Tensor

def compute_ect_point_cloud(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
) -> Tensor:
    """
    Computes the ECT of a point cloud.
    """
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v.T).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    ect = ect.permute(0, 2, 1)

    return 2 * (ect / torch.amax(ect, dim=(-1, -2), keepdim=True).clamp(min=1e-6)) - 1
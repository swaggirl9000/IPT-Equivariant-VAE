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

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions. In our case, this is 
        the Lebedev directions. 
    radius : float
        Radius of the interval to discretize the ECT into. (Is irrelevant for
        this experiment.)
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,D,R] where B is the number of
        point clouds (thus ECT's), D is the number of direction and R is the
        resolution.
    """
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v.T).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    ect = ect.permute(0, 2, 1)

    return 2 * (ect / torch.amax(ect, dim=(-1, -2), keepdim=True).clamp(min=1e-6)) - 1
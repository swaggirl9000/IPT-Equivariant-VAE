import torch
from e3nn import o3

def compute_pointwise_coefficients(pc: torch.Tensor, l_max: int = 9, R: int = 32):
    """
    Compute the pointwise coefficients of the Zernike polynomials for a given point cloud.

    Args:
        pc (torch.Tensor): The input point cloud of shape (B, N, 3).
        l_max (int): The maximum degree of the Zernike polynomials.
        R (int): The number of radial samples.  
    """
    B, N, _ = pc.shape
    
    norms = pc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_pc = pc/norms  
    irreps_sh = o3.Irreps.spherical_harmonics(l_max)
    Y = o3.spherical_harmonics(
        irreps_sh, unit_pc, normalize=True, normalization="integral"
    )
    radii      = norms.squeeze(-1)                         
    thresholds = torch.linspace(0.0, 1.0, R, device=pc.device).view(1, 1, R)
    sigma = 2.0 / R
    radial_weights = torch.exp(
        -((radii.unsqueeze(-1) - thresholds) ** 2) / (2 * sigma ** 2)
    )
    
    coeffs = torch.einsum('bnf, bnr -> bfr', Y, radial_weights)
    
    return coeffs
    
    
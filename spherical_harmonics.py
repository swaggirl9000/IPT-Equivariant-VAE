import torch
from torch import Tensor
from e3nn import o3
from torch import nn

class SphericalHarmonicProjection(nn.Module):
    """
    Projects a function sampled on the Lebedev grid onto SH coefficients.

    Input : f  (B, D, R)  — ECT image, D = num Lebedev directions
    Output: c  (B, C, R)  — SH coefficients, C = (l_max+1)²
    """
    def __init__(self, dirs: Tensor, weights: Tensor, l_max: int=9):
        super().__init__()
        self.l_max = l_max
        self.register_buffer("dirs", dirs)      
        self.register_buffer("weights", weights)
        
        irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        Y = o3.spherical_harmonics(
            irreps_sh, dirs, normalize=True, normalization="integral"
        )   
        self.register_buffer("Y", Y)

    def forward(self, f: Tensor) -> Tensor:

        w = self.weights                      
        Y = self.Y                              
        wY = (w.unsqueeze(1) * Y)            
        c = torch.einsum("bdr, dc -> bcr", f, wY)
        return c  
    
class InverseSphericalHarmonicProjection(nn.Module):
    """
    Reconstructs the ECT image on the Lebedev grid from SH coefficients.

    Input : c     (B, C, R)  — SH coefficients
    Output: f_hat (B, D, R)  — reconstructed ECT image on grid
    """
    def __init__(self, dirs: Tensor, l_max: int = 9):
        super().__init__()
        irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        Y = o3.spherical_harmonics(
            irreps_sh, dirs, normalize=True, normalization="integral"
        )  
        self.register_buffer("Y", Y)

    def forward(self, c: Tensor) -> Tensor:
        f_hat = torch.einsum("bcr, dc -> bdr", c, self.Y) 
        return f_hat
import math
import torch
from torch import Tensor
from e3nn import o3
from torch import nn

class SphericalHarmonicProjection(nn.Module):
    """
    Projects a function sampled on the Lebedev grid onto SH coefficients.
    """
    def __init__(self, dirs: Tensor, weights: Tensor, l_max: int=9):
        super().__init__()
        self.l_max = l_max
        if torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-3):
            weights = weights * 4 * math.pi
            
        self.register_buffer("dirs", dirs)      
        self.register_buffer("weights", weights)
        
        irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        Y = o3.spherical_harmonics(
            irreps_sh, self.dirs, normalize=True, normalization="integral"
        )   
        self.register_buffer("Y", Y)

    def forward(self, f: Tensor) -> Tensor:
        if f.dim() == 4 and f.shape[1] == 1:
            f = f.squeeze(1) 
            
        if f.shape[1] != self.dirs.shape[0]:
            if f.shape[2] == self.dirs.shape[0]:
                f = f.movedim(2, 1) 
            else:
                raise ValueError(f"Input shape {f.shape} doesn't match grid size {self.dirs.shape[0]}")

        wY = (self.weights.unsqueeze(1) * self.Y)            
        c = torch.einsum("bdr, dc -> bcr", f, wY)
        
        return c.movedim(1, 2) 


class InverseSphericalHarmonicProjection(nn.Module):
    """
    Reconstructs the ECT image on the Lebedev grid from SH coefficients.
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
import torch
from torch import nn, Tensor
from e3nn import o3


class SphericalHarmonicProjection(nn.Module):
    def __init__(self, dirs: Tensor, weights: Tensor, l_max: int) -> None:
        super().__init__()
        self.irreps = o3.Irreps.spherical_harmonics(l_max)

        dirs_norm = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        sh_basis = o3.spherical_harmonics(
            self.irreps,
            dirs_norm,
            normalize=True,
            normalization="integral",
        )  

        weighted_basis = weights.unsqueeze(-1) * sh_basis  

        self.register_buffer("weighted_basis", weighted_basis)

    def forward(self, ect: Tensor) -> Tensor:
        """
        ect: (B, num_dirs, R)  ->  sh_features: (B, R, D_sh)
        """
        assert ect.dim() == 3, (
            f"Expected ect of shape [B, num_dirs, R], got {tuple(ect.shape)}"
        )
        B, num_dirs, R = ect.shape
        assert num_dirs == self.weighted_basis.shape[0], (
            f"Direction mismatch: ect has {num_dirs} dirs but basis has "
            f"{self.weighted_basis.shape[0]}"
        )

        sh_features = torch.einsum("bir, id -> brd", ect, self.weighted_basis)
        return sh_features


class InverseSphericalHarmonicProjection(nn.Module):
    def __init__(self, dirs: Tensor, l_max: int = 9):
        super().__init__()
        irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        Y = o3.spherical_harmonics(
            irreps_sh,
            dirs,
            normalize=True,
            normalization="integral",
        )  # (num_dirs, D_sh)
        self.register_buffer("Y", Y)

    def forward(self, c: Tensor) -> Tensor:
        f_hat = torch.einsum("brd, id -> bir", c, self.Y)
        return f_hat
import math
import torch
from e3nn import o3
from functools import lru_cache
from typing import List

@lru_cache(maxsize=None)
def _poly_coeffs(n: int, l: int) -> List[float]:
    assert n >= l >= 0 and (n - l) % 2 == 0, f"Invalid Zernike indices n={n}, l={l}"
    p = (n - l) // 2
    return [
        float(
            ((-1) ** s)
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial(p - s)
                * math.factorial((n + l) // 2 - s + 1)
            )
        )
        for s in range(p + 1)
    ]


def _eval_raw(n: int, l: int, r: torch.Tensor) -> torch.Tensor:
    coeffs = _poly_coeffs(n, l)
    return sum(c * r.pow(n - 2 * s) for s, c in enumerate(coeffs))


@lru_cache(maxsize=None)
def _norm_factor(n: int, l: int, n_pts: int = 2048) -> float:
    r        = torch.linspace(0.0, 1.0, n_pts)   # CPU tensor
    val      = _eval_raw(n, l, r)
    integrand = val ** 2 * r ** 2
    dr       = 1.0 / (n_pts - 1)
    norm_sq  = float((integrand[:-1] + integrand[1:]).sum() * dr / 2.0)
    return 1.0 / math.sqrt(max(norm_sq, 1e-12))


def eval_zernike_radial(n: int, l: int, r: torch.Tensor) -> torch.Tensor:
    return _eval_raw(n, l, r) * _norm_factor(n, l)

def compute_pointwise_coefficients(
    pc:    torch.Tensor,
    l_max: int = 9,
    R:     int = 32,
) -> torch.Tensor:
    """
    Compute the zernikegram of a point cloud: a 3D Zernike expansion using
    real spherical harmonics (angular) × Zernike radial polynomials (radial).
    """
    B, N, _ = pc.shape

    # Unit directions and clamped radii
    norms   = pc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_pc = pc / norms                                    
    radii   = norms.squeeze(-1).clamp(max=1.0)          

    irreps_sh = o3.Irreps.spherical_harmonics(l_max)
    Y = o3.spherical_harmonics(
        irreps_sh, unit_pc, normalize=True, normalization="integral"
    )

    coeffs = torch.zeros(B, (l_max + 1) ** 2, R, device=pc.device, dtype=pc.dtype)

    sh_idx = 0
    for l in range(l_max + 1):
        m   = 2 * l + 1
        Y_l = Y[:, :, sh_idx : sh_idx + m]               

        for k in range(R):
            n   = l + 2 * k                                 # radial order
            R_nl = eval_zernike_radial(n, l, radii)       
            coeffs[:, sh_idx : sh_idx + m, k] = torch.einsum(
                "bn, bnm -> bm", R_nl, Y_l
            )

        sh_idx += m

    return coeffs

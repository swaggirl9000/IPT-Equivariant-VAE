# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from get_zernikegrams import compute_pointwise_coefficients
# from get_ipt import compute_ect_point_cloud
# from spherical_harmonics import SphericalHarmonicProjection
# from vae import EquivariantVAE
# from equivariant_decoder import EquivariantDecoder


# class IPTVAEPipeline(nn.Module):
#     def __init__(self,
#                  dirs: torch.Tensor,
#                  weights: torch.Tensor,
#                  l_max: int = 6,
#                  R: int = 8):
#         super().__init__()
#         self.l_max = l_max
#         self.R = R
#         self.register_buffer("dirs", dirs)

#         handoff_parts = []
#         for l in range(l_max + 1):
#             p = 'e' if l % 2 == 0 else 'o'
#             handoff_parts.append(f"16x{l}{p}")
#         vae_handoff_str = " + ".join(handoff_parts)

#         self.sft = SphericalHarmonicProjection(dirs, weights, l_max=l_max)

#         self.vae = EquivariantVAE(
#             l_max=l_max,
#             R=R,
#             vae_out_irreps_str=vae_handoff_str,
#             hidden_mul=64,
#             latent_channels=16,
#         )

#         self.decoder = EquivariantDecoder(
#             vae_out_irreps_str=vae_handoff_str,
#             l_max=l_max,
#             R=R,
#             hidden_mul=64,
#         )

#     def forward(self, pc: torch.Tensor):
#         # PC -> ECT grid
#         f_spatial = compute_ect_point_cloud(
#             pc, self.dirs, radius=1.0, resolution=self.R, scale=10.0
#         )

#         # ECT grid -> SH coefficients
#         c_ipt_sh = self.sft(f_spatial)

#         # PC -> Zernike coefficients (reconstruction target)
#         c_zernike = compute_pointwise_coefficients(pc, l_max=self.l_max, R=self.R)

#         # SH coefficients -> VAE -> v_raw
#         v_raw, mu, logvar_expanded = self.vae(c_ipt_sh)

#         # v_raw -> Equivariant decoder -> predicted Zernike-space vector
#         c_pred = self.decoder(v_raw)

#         return c_pred, c_zernike, mu, logvar_expanded


# def compute_loss(
#     c_pred:          torch.Tensor,
#     c_zernike:       torch.Tensor,
#     mu:              torch.Tensor,
#     logvar_expanded: torch.Tensor,
#     l_max:           int,
#     beta:            float = 0.0,
# ) -> dict:
#     """
#     L_zernike   per-l cosine loss: Zernike(PC) vs equivariant decoder output
#     L_kl        KL divergence of the VAE posterior from N(0, I)
#                 weighted by beta — use a warmup schedule in train.py

#     Total = L_zernike + beta * L_kl
#     """
#     B, _, _ = c_pred.shape
#     zernike_terms = []
#     sh_idx = 0

#     for l in range(l_max + 1):
#         m = 2 * l + 1
#         a = c_pred[:, sh_idx:sh_idx + m, :].reshape(B, -1)
#         b = c_zernike[:, sh_idx:sh_idx + m, :].reshape(B, -1)
#         zernike_terms.append(
#             (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
#         )
#         sh_idx += m

#     L_zernike = torch.stack(zernike_terms).mean()

#     logvar_clamped = logvar_expanded.clamp(-10, 10)
#     L_kl = -0.5 * (
#         1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
#     ).sum(dim=-1).mean()

#     total = L_zernike + beta * L_kl

#     return dict(
#         loss      = total,
#         L_zernike = L_zernike.detach(),
#         L_kl      = L_kl.detach(),
#     )

import torch
import torch.nn as nn
import torch.nn.functional as F

from get_zernikegrams import compute_pointwise_coefficients
from get_ipt import compute_ect_point_cloud
from spherical_harmonics import SphericalHarmonicProjection
from vae import EquivariantVAE
from equivariant_decoder import EquivariantDecoder


class IPTVAEPipeline(nn.Module):
    def __init__(self,
                 dirs: torch.Tensor,
                 weights: torch.Tensor,
                 l_max: int = 6,
                 R: int = 8):
        super().__init__()
        self.l_max = l_max
        self.R = R
        self.register_buffer("dirs", dirs)

        handoff_parts = []
        for l in range(l_max + 1):
            p = 'e' if l % 2 == 0 else 'o'
            handoff_parts.append(f"16x{l}{p}")
        vae_handoff_str = " + ".join(handoff_parts)

        self.sft = SphericalHarmonicProjection(dirs, weights, l_max=l_max)

        self.vae = EquivariantVAE(
            l_max=l_max,
            R=R,
            vae_out_irreps_str=vae_handoff_str,
            hidden_mul=64,
            latent_channels=16,
        )

        self.decoder = EquivariantDecoder(
            vae_out_irreps_str=vae_handoff_str,
            l_max=l_max,
            R=R,
            hidden_mul=64,
        )

    def forward(self, pc: torch.Tensor):
        # PC -> ECT grid
        f_spatial = compute_ect_point_cloud(
            pc, self.dirs, radius=1.0, resolution=self.R, scale=10.0
        )

        # ECT grid -> SH coefficients
        c_ipt_sh = self.sft(f_spatial)

        # PC -> Zernike coefficients (reconstruction target)
        c_zernike = compute_pointwise_coefficients(pc, l_max=self.l_max, R=self.R)

        # SH coefficients -> VAE -> v_raw
        v_raw, mu, logvar_expanded = self.vae(c_ipt_sh)

        # v_raw -> Equivariant decoder -> predicted Zernike-space vector
        c_pred = self.decoder(v_raw)

        return c_pred, c_zernike, mu, logvar_expanded


def compute_loss(
    c_pred:          torch.Tensor,
    c_zernike:       torch.Tensor,
    mu:              torch.Tensor,
    logvar_expanded: torch.Tensor,
    l_max:           int,
    beta:            float = 0.0,
) -> dict:
    """
    L_zernike   per-l cosine loss: Zernike(PC) vs equivariant decoder output
    L_kl        KL divergence of the VAE posterior from N(0, I)
                weighted by beta — use a warmup schedule in train.py

    Total = L_zernike + beta * L_kl
    """
    B, _, _ = c_pred.shape

    # Flatten all SH coefficients and radial channels into a single vector
    # per batch element so every coefficient is weighted equally.
    a = c_pred.reshape(B, -1)
    b = c_zernike.reshape(B, -1)
    L_zernike = (1.0 - F.cosine_similarity(a, b, dim=1)).mean()

    logvar_clamped = logvar_expanded.clamp(-10, 10)
    L_kl = -0.5 * (
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
    ).sum(dim=-1).mean()

    total = L_zernike + beta * L_kl

    return dict(
        loss      = total,
        L_zernike = L_zernike.detach(),
        L_kl      = L_kl.detach(),
    )
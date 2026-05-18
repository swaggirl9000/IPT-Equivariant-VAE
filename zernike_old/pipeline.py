import torch
import torch.nn as nn
import torch.nn.functional as F

from get_zernikegrams import compute_pointwise_coefficients
from get_ipt import compute_ect_point_cloud
from spherical_harmonics import SphericalHarmonicProjection, InverseSphericalHarmonicProjection
from vae import EquivariantVAE
from equivariant_decoder import EquivariantDecoder


class IPTVAEPipeline(nn.Module):
    """
    SO(3)-Equivariant VAE for the Inner Product Transform.

    Pipeline
    --------
    PC ──► ECT/IPT ──► SH projection ──► EquivariantVAE ──► ISH ──► EquivariantDecoder ──► c_pred
                       (spherical FT)         │                                          
                                              │
                                              └──► c_recon   [IPT-SH loss target]

    Two parallel decoder outputs from the VAE
    ------------------------------------------
    1. v_raw (flat steerable vector, handoff irreps)
       → EquivariantDecoder → c_pred (B, F, R)
       Used for Zernike loss against the direct Zernike transform of the input PC.

    2. c_recon (B, F, R, same space as c_ipt_sh)
       → InverseSphericalHarmonicProjection → f_recon (B, D, R)
       c_recon is also compared to c_ipt_sh for the IPT-SH reconstruction loss.
       f_recon is returned for downstream use (visualisation, analysis).

    Loss
    ----
    L_zernike  : per-l cosine similarity between c_pred and c_zernike
                 (c_zernike = direct Zernike expansion of the input PC)
    L_ipt_sh   : per-l cosine similarity between c_recon and c_ipt_sh
                 (VAE reconstruction quality in the SH-coefficient space)
    L_kl       : KL divergence, normalised by latent dimension

    Total = L_zernike + L_ipt_sh + beta * L_kl

    Equivariance guarantee
    ----------------------
    ECT + SH projection : equivariant by construction (Wigner-D on SH coefficients)
    EquivariantVAE       : e3nn E3Linear + Gate throughout
    EquivariantDecoder   : e3nn E3Linear + Gate throughout
    ISH                  : linear in SH basis → equivariant
    → full pipeline is SO(3)-equivariant; verified in equivariance_test.py
    """

    def __init__(
        self,
        dirs:    torch.Tensor,
        weights: torch.Tensor,
        l_max:   int = 6,
        R:       int = 8,
    ):
        super().__init__()
        self.l_max = l_max
        self.R     = R
        self.register_buffer("dirs", dirs)

        # Handoff irreps between VAE decoder and EquivariantDecoder
        handoff_parts   = [f"16x{l}{'e' if l%2==0 else 'o'}" for l in range(l_max + 1)]
        vae_handoff_str = " + ".join(handoff_parts)

        self.sft = SphericalHarmonicProjection(dirs, weights, l_max=l_max)
        self.ish = InverseSphericalHarmonicProjection(dirs, l_max=l_max)

        self.vae = EquivariantVAE(
            l_max              = l_max,
            R                  = R,
            vae_out_irreps_str = vae_handoff_str,
            hidden_mul         = 64,
            latent_channels    = 16,
        )

        self.decoder = EquivariantDecoder(
            vae_out_irreps_str = vae_handoff_str,
            l_max              = l_max,
            R                  = R,
            hidden_mul         = 64,
        )

    def forward(self, pc: torch.Tensor):
        # PC --> ECT (IPT)                                    
        f_spatial = compute_ect_point_cloud(
            pc, self.dirs, radius=1.0, resolution=self.R, scale=10.0
        )

        # ECT --> SH coefficients  
        c_ipt_sh = self.sft(f_spatial)

        # Zernike target 
        c_zernike = compute_pointwise_coefficients(pc, l_max=self.l_max, R=self.R)

        # Equivariant VAE
        v_raw, c_recon, mu, logvar_expanded = self.vae(c_ipt_sh)

        # ISH
        f_recon = self.ish(c_recon)                          
        _ = f_recon  

        # Equivariant Decoder
        c_pred = self.decoder(v_raw)               

        return c_pred, c_zernike, c_ipt_sh, c_recon, mu, logvar_expanded

def compute_loss(
    c_pred:          torch.Tensor,
    c_zernike:       torch.Tensor,
    c_ipt_sh:        torch.Tensor,
    c_recon:         torch.Tensor,
    mu:              torch.Tensor,
    logvar_expanded: torch.Tensor,
    l_max:           int,
    beta:            float = 0.0,
) -> dict:
    """
    Three-term loss:

    L_zernike  per-l cosine loss between c_pred and c_zernike
               (decoder output vs direct Zernike expansion of the input PC)

    L_ipt_sh   per-l cosine loss between c_recon and c_ipt_sh
               (VAE SH reconstruction vs SH projection of the ECT input)

    L_kl       KL divergence normalised by latent dimension

    Total loss = L_zernike + L_ipt_sh + beta * L_kl


    """
    B = c_pred.shape[0]

    # def _per_l_cosine(a_full: torch.Tensor, b_full: torch.Tensor) -> torch.Tensor:
    #     terms  = []
    #     sh_idx = 0
    #     for l in range(l_max + 1):
    #         m = 2 * l + 1
    #         a = a_full[:, sh_idx:sh_idx + m, :].reshape(B, -1)
    #         b = b_full[:, sh_idx:sh_idx + m, :].reshape(B, -1)
    #         terms.append((1.0 - F.cosine_similarity(a, b, dim=-1)).mean())
    #         sh_idx += m
    #     return torch.stack(terms).mean()
    def _per_l_cosine(a_full: torch.Tensor, b_full: torch.Tensor) -> torch.Tensor:
        """Average cosine loss over all l blocks, weighted by inverse signal magnitude.
        Prevents strong even-l degrees from dominating over weak odd-l degrees."""
        terms   = []
        weights = []
        sh_idx  = 0
        for l in range(l_max + 1):
            m = 2 * l + 1
            a = a_full[:, sh_idx:sh_idx + m, :].reshape(B, -1)
            b = b_full[:, sh_idx:sh_idx + m, :].reshape(B, -1)
            signal = b.norm(dim=-1).mean().detach().clamp(min=1e-6)
            w = 1.0 / signal
            terms.append((1.0 - F.cosine_similarity(a, b, dim=-1)).mean() * w)
            weights.append(w)
            sh_idx += m
        return torch.stack(terms).sum() / torch.stack(weights).sum()

    # Zernike loss: decoder output vs direct Zernike expansion of the input PC
    L_zernike = _per_l_cosine(c_pred, c_zernike)

    # IPT-SH loss: VAE SH reconstruction vs SH projection of the ECT input
    L_ipt_sh  = _per_l_cosine(c_recon, c_ipt_sh)

    # KL divergence, normalised by latent dimension
    logvar_clamped = logvar_expanded.clamp(-4, 4)
    L_kl = -0.5 * (
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
    ).sum(dim=-1).mean()
    L_kl = L_kl / logvar_expanded.shape[-1]

    total = L_zernike + L_ipt_sh + beta * L_kl

    return dict(
        loss      = total,
        L_zernike = L_zernike.detach(),
        L_ipt_sh  = L_ipt_sh.detach(),
        L_kl      = L_kl.detach(),
    )


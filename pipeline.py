import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from get_zernikegrams import compute_pointwise_coefficients
from get_directions import get_directions
from get_ipt import compute_ect_point_cloud
from spherical_harmonics import SphericalHarmonicProjection
from vae import EquivariantVAE
from equivariant_decoder import EquivariantDecoder
from get_mnist import PointCloudMNIST

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
            latent_channels=16
        )
        
        self.decoder = EquivariantDecoder(
            vae_out_irreps_str=vae_handoff_str,
            l_max=l_max,
            R=R,
            hidden_mul=64
        )
        
    def forward(self, pc: torch.Tensor):
        # PC -> IPT Grid 
        f_spatial = compute_ect_point_cloud(pc, self.dirs, radius=1.0, resolution=self.R, scale=10.0)
        
        # IPT Grid -> Spherical Harmonics 
        c_ipt_sh = self.sft(f_spatial) 
        
        c_zernike = compute_pointwise_coefficients(pc, l_max=self.l_max, R=self.R)
        
        # Target -> VAE -> VAE Output Vector
        v_raw, c_vae_out, mu, logvar = self.vae(c_ipt_sh)
        
        # VAE Output Vector -> Equivariant Decoder 
        c_pred = self.decoder(v_raw) 
        
        return c_pred, c_zernike, c_ipt_sh, c_vae_out, mu, logvar
    
def compute_loss(
    c_pred:    torch.Tensor,  
    c_zernike: torch.Tensor,   
    c_ipt_sh:  torch.Tensor, 
    c_vae_out: torch.Tensor,  
    mu:        torch.Tensor, 
    logvar:    torch.Tensor,  
    l_max:     int,
    beta:      float = 0.01,
    lambda_ipt: float = 1.0,
) -> dict:
    """
    L_zernike   cosine loss between Zernike(PC) and equivariant decoder output

    L_ipt_sh    MSE between c_ipt_sh (VAE input) and c_vae_out (VAE reconstruction)

    L_kl        KL divergence of the VAE posterior from the standard normal prior

    Total = L_zernike + lambda_ipt * L_ipt_sh + beta * L_kl
    """
    B, num_sh, R = c_pred.shape
    zernike_terms = []
    sh_idx = 0
    
    for l in range(l_max + 1):
        m = 2 * l + 1
        a = c_pred[:, sh_idx:sh_idx+m, :].reshape(B, -1)
        b = c_zernike[:, sh_idx:sh_idx+m, :].reshape(B, -1)
        zernike_terms.append(
            (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
        )
        sh_idx += m
        
    L_zernike = torch.stack(zernike_terms).mean()
    L_ipt_sh = F.mse_loss(c_vae_out, c_ipt_sh)
    L_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    total = L_zernike + lambda_ipt * L_ipt_sh + beta * L_kl

    return dict(
        loss      = total,
        L_zernike = L_zernike.detach(),
        L_ipt_sh  = L_ipt_sh.detach(),
        L_kl      = L_kl.detach(),
    )
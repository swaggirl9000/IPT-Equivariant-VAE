import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
import e3nn.nn as enn
from e3nn.o3 import Linear as E3Linear
from e3nn.o3 import Irreps

class EquivariantVAE(nn.Module):
    """
    A pure SO(3) Equivariant Variational Autoencoder.
    Bridges the gap between the SFT output and the EquivariantDecoder.
    """
    def __init__(
                self,
                l_max: int,
                R: int,                   
                vae_out_irreps_str: str,  
                hidden_mul: int = 64, 
                latent_channels: int = 16  
                ):
        super().__init__()
        self.l_max = l_max
        self.R     = R
        self.F     = (l_max + 1) ** 2
        self.latent_channels = latent_channels
        
        input_parts = []
        for l in range(l_max + 1):
            p = 'e' if l % 2 == 0 else 'o'
            input_parts.append(f"{R}x{l}{p}")
        self.input_irreps = Irreps(" + ".join(input_parts))

        hidden_parts = []
        for l in range(l_max + 1):
            mul = 64 if l <= 2 else 32 
            p = 'e' if l % 2 == 0 else 'o'
            hidden_parts.append(f"{mul}x{l}{p}")
        self.hidden_irreps = Irreps(" + ".join(hidden_parts))

        latent_parts = []
        for l in range(l_max + 1):
            p = 'e' if l % 2 == 0 else 'o'
            latent_parts.append(f"{latent_channels}x{l}{p}") 
            
        latent_str = " + ".join(latent_parts)
        self.latent_irreps = Irreps(latent_str)
        
        num_mu_channels = latent_channels * (l_max + 1)
        self.logvar_irreps = Irreps(f"{num_mu_channels}x0e")
        
        self.out_irreps = Irreps(vae_out_irreps_str)

        scalars_irreps = Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        gated_irreps   = Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0])
        n_gates        = sum(mul for mul, ir in self.hidden_irreps if ir.l > 0)
        gates_irreps   = Irreps(f"{n_gates}x0e")

        self.pre_gate_irreps = scalars_irreps + gates_irreps + gated_irreps

        self.gate = enn.Gate(
            irreps_scalars = scalars_irreps,
            act_scalars    = [F.silu] * len(scalars_irreps),
            irreps_gates   = gates_irreps,
            act_gates      = [torch.sigmoid] * len(gates_irreps),
            irreps_gated   = gated_irreps,
        )

        self.enc_lin1 = E3Linear(self.input_irreps, self.pre_gate_irreps)
        self.enc_lin2 = E3Linear(self.hidden_irreps, self.pre_gate_irreps)
        self.enc_lin3 = E3Linear(self.hidden_irreps, self.pre_gate_irreps)

        self.enc_mu     = E3Linear(self.hidden_irreps, self.latent_irreps)
        self.enc_logvar = E3Linear(self.hidden_irreps, self.logvar_irreps)

        self.dec_lin1 = E3Linear(self.latent_irreps,  self.pre_gate_irreps)
        self.dec_lin2 = E3Linear(self.hidden_irreps,  self.pre_gate_irreps)
        self.dec_lin3 = E3Linear(self.hidden_irreps,  self.pre_gate_irreps)
        self.dec_out  = E3Linear(self.hidden_irreps,  self.out_irreps)
        
        self.recon_lin1 = E3Linear(self.hidden_irreps, self.pre_gate_irreps)
        self.recon_out  = E3Linear(self.hidden_irreps, self.input_irreps)

    def _gated_layer(self, linear: E3Linear, x: torch.Tensor) -> torch.Tensor:
        return self.gate(linear(x))

    def _expand_scalars(self, x: torch.Tensor) -> torch.Tensor:
        """Expands (B, 112) scalars to (B, 784) to match mu's geometric dimensions."""
        expanded = []
        idx = 0
        for l in range(self.l_max + 1):
            m = 2 * l + 1
            x_slice = x[:, idx : idx + self.latent_channels]
            expanded.append(x_slice.repeat_interleave(m, dim=1))
            idx += self.latent_channels
        return torch.cat(expanded, dim=1)
    
    def reparameterize(self, mu: torch.Tensor, logvar_expanded: torch.Tensor) -> torch.Tensor:
        if self.training:
            std_expanded = (0.5 * logvar_expanded).exp()
            return mu + std_expanded * torch.randn_like(mu)
        return mu
    
    def _c_to_e3nn(self, c: torch.Tensor) -> torch.Tensor:
        B = c.shape[0]
        segments = []
        sh_idx = 0
        for l in range(self.l_max + 1):
            m = 2 * l + 1
            seg = c[:, sh_idx : sh_idx + m, :]
            seg = seg.permute(0, 2, 1).reshape(B, -1)
            segments.append(seg)
            sh_idx += m
        return torch.cat(segments, dim=-1)   

    def _e3nn_to_c(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        segments = []
        sh_idx = 0
        for l in range(self.l_max + 1):
            m = 2 * l + 1
            block_size = self.R * m
            seg = x[:, sh_idx : sh_idx + block_size]
            seg = seg.reshape(B, self.R, m).permute(0, 2, 1)
            segments.append(seg)
            sh_idx += block_size
        return torch.cat(segments, dim=1)

    def forward(self, c_target: torch.Tensor):
        B = c_target.shape[0]

        x = self._c_to_e3nn(c_target)  

        x = self._gated_layer(self.enc_lin1, x)
        x = self._gated_layer(self.enc_lin2, x)
        x = self._gated_layer(self.enc_lin3, x)
        
        mu     = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        
        logvar_expanded = self._expand_scalars(logvar)
        
        z = self.reparameterize(mu, logvar_expanded)
        
        h     = self._gated_layer(self.dec_lin1, z)
        h     = self._gated_layer(self.dec_lin2, h)
        h     = self._gated_layer(self.dec_lin3, h)
        v_raw = self.dec_out(h)

        r     = self._gated_layer(self.recon_lin1, h)
        r_flat = self.recon_out(r)                    
        c_vae_out = self._e3nn_to_c(r_flat)     

        return v_raw, c_vae_out, mu, logvar_expanded
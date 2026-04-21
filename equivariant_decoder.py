import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
import e3nn.nn as enn
from e3nn.o3 import Linear as E3Linear
from e3nn.o3 import Irreps


class EquivariantDecoder(nn.Module):
    """
    Maps the H-VAE decoder output vector back into the same
    spherical Fourier space as the ZFT input.

    Input:  flat steerable vector  
    Output: SH coefficient tensor of shape (B, F, R)
              where F = (l_max+1)^2, matching compute_pointwise_coefficients output.

    This output is compared directly against the ZFT of the input point cloud.
    """

    def __init__(
        self,
        vae_out_irreps_str: str,  
        l_max: int = 6,    
        R: int = 1,         
        hidden_mul: int = 64,     
    ):
        super().__init__()
        self.l_max = l_max
        self.R = R
        self.F = (l_max + 1) ** 2

        self.input_irreps = Irreps(vae_out_irreps_str)

        hidden_parts = []
        for l in range(l_max + 1):
            mul = 64 if l <= 2 else 32 
            p = 'e' if l % 2 == 0 else 'o'
            hidden_parts.append(f"{mul}x{l}{p}")
        self.hidden_irreps = Irreps(" + ".join(hidden_parts))
        
        output_parts = [
            f"{R}x{l}{'e' if l % 2 == 0 else 'o'}"
            for l in range(l_max + 1)
        ]
        self.output_irreps = Irreps(" + ".join(output_parts))

        scalars_irreps = Irreps(
            [(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0]
        )
        gated_irreps = Irreps(
            [(mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0]
        )
        n_gates      = sum(mul for mul, ir in self.hidden_irreps if ir.l > 0)
        gates_irreps = Irreps(f"{n_gates}x0e")

        self.pre_gate_irreps = scalars_irreps + gates_irreps + gated_irreps

        self.gate = enn.Gate(
            irreps_scalars = scalars_irreps,
            act_scalars    = [F.silu] * len(scalars_irreps),
            irreps_gates   = gates_irreps,
            act_gates      = [torch.sigmoid] * len(gates_irreps),
            irreps_gated   = gated_irreps,
        )

        self.lin1 = E3Linear(self.input_irreps,  self.pre_gate_irreps)
        self.lin2 = E3Linear(self.hidden_irreps, self.pre_gate_irreps)
        self.lin3 = E3Linear(self.hidden_irreps, self.pre_gate_irreps)
        self.lin4 = E3Linear(self.hidden_irreps, self.output_irreps)

    def _gated_layer(self, linear: E3Linear, x: torch.Tensor) -> torch.Tensor:
        return self.gate(linear(x))

    def forward(self, v_raw: torch.Tensor) -> torch.Tensor:
        B = v_raw.shape[0]
        x = self._gated_layer(self.lin1, v_raw)
        x = self._gated_layer(self.lin2, x)
        x = self._gated_layer(self.lin3, x)
        c_flat = self.lin4(x)   

        segments = []
        sh_idx = 0
        for l in range(self.l_max + 1):
            m = 2 * l + 1         
            block_size = self.R * m
            
            seg = c_flat[:, sh_idx : sh_idx + block_size]
            seg = seg.reshape(B, self.R, m).permute(0, 2, 1)
            segments.append(seg)
            sh_idx += block_size

        c_out = torch.cat(segments, dim=1)  
        return c_out
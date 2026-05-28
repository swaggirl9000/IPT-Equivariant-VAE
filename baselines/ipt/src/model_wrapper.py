"""
A wrapper to give our model the same signature as the
one in pointflow and make it accept the same type of data.
"""

import torch

# from models.vae_baseline import BaseLightningModel as VAE
from scipy import sparse

from layers.directions import generate_uniform_directions
from layers.ect import EctLayer, compute_ect_point_cloud
from models.encoder_new import BaseLightningModel as Encoder

DEVICE = "cuda:0"


def normalize(pts):
    assert pts.shape[1:] == (2048, 3)
    pts_means = pts.mean(axis=-2, keepdim=True)
    pts = pts - pts_means
    pts_norms = torch.norm(pts, dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    pts = pts / pts_norms
    return pts, pts_means, pts_norms


class ModelWrapper:
    def __init__(self, encoder: Encoder, vae: None = None) -> None:
        self.encoder = encoder
        self.encoder.eval()
        self.vae = vae
        if vae is not None:
            self.vae.model.eval()

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        out_pc, sample_ect = model.sample(len(batch), pc_shape)
        The way we expect the input
        B is the number of point clouds
        N is the number of points per cloud.
        _, out_pc = model.sample(B, N)
        """
        ect_samples = self.vae.model.sample(n=num_samples)

        vae_pointcloud = self.encoder(ect_samples).view(
            -1,
            self.encoder.config.num_pts,
            3,
        )
        return vae_pointcloud, ect_samples

    @torch.no_grad()
    def reconstruct(self, ect):
        if self.vae is not None:
            output = self.vae.model(ect)
            pointcloud = self.encoder(output[0].squeeze()).view(
                -1, self.encoder.config.num_pts, 3
            )
            reconstructed_ect = output[0]
        else:
            pointcloud = self.encoder(ect.squeeze()).view(
                -1, self.encoder.config.num_pts, 3
            )
            reconstructed_ect = ect

        return pointcloud, reconstructed_ect

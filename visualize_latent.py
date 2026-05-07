"""
Latent space visualisation.

Encodes N test samples, collects their mu vectors, runs PCA and optionally
UMAP, plots coloured by digit class label.

Usage:
    python visualize_latent.py --checkpoint checkpoint_lmax2_leb19.pt --l_max 2 --lebedev 19
"""

import argparse
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from get_directions import get_directions
from get_mnist import PointCloudMNIST
from pipeline import IPTVAEPipeline
from get_ipt import compute_ect_point_cloud

@torch.no_grad()
def collect_latents(
    model:      IPTVAEPipeline,
    dataset:    torch.utils.data.Dataset,
    n_samples:  int = 500,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode n_samples from the dataset, return (mu_vectors, labels).
    """
    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mus    = []
    all_labels = []
    collected  = 0

    for pc, label in loader:
        if collected >= n_samples:
            break
        pc = pc.to(device)

        f_spatial = compute_ect_point_cloud(
            pc, model.dirs, radius=1.0, resolution=model.R, scale=10.0
        )
        c_ipt_sh = model.sft(f_spatial)
        x        = model.vae._c_to_e3nn(c_ipt_sh)
        x        = model.vae._gated_layer(model.vae.enc_lin1, x)
        x        = model.vae._gated_layer(model.vae.enc_lin2, x)
        x        = model.vae._gated_layer(model.vae.enc_lin3, x)
        mu       = model.vae.enc_mu(x)  

        all_mus.append(mu.cpu().numpy())
        all_labels.append(label.numpy())
        collected += pc.shape[0]

    mus    = np.concatenate(all_mus,    axis=0)[:n_samples]
    labels = np.concatenate(all_labels, axis=0)[:n_samples]
    return mus, labels


def plot_pca(mus, labels, save_path=None):
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(mus)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = cm.tab10(np.linspace(0, 1, 10))

    for cls in range(10):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.6, color=colours[cls], label=str(cls)
        )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("Latent space PCA — coloured by digit class")
    ax.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA plot to {save_path}")
    else:
        plt.show()

    return coords, var


def plot_umap(mus, labels, save_path=None):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords  = reducer.fit_transform(mus)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = cm.tab10(np.linspace(0, 1, 10))

    for cls in range(10):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.6, color=colours[cls], label=str(cls)
        )

    ax.set_title("Latent space UMAP — coloured by digit class")
    ax.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved UMAP plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--l_max",      type=int, default=2)
    parser.add_argument("--lebedev",    type=int, default=19)
    parser.add_argument("--R",          type=int, default=8)
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--save_dir",   type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dirs, weights = get_directions(args.lebedev)
    dirs, weights = dirs.to(device), weights.to(device)

    model = IPTVAEPipeline(dirs, weights, l_max=args.l_max, R=args.R).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False), strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset = PointCloudMNIST(root="./data", train=False, num_points=256)

    print(f"Collecting latents for {args.n_samples} samples...")
    mus, labels = collect_latents(model, dataset, n_samples=args.n_samples)
    print(f"Collected: mus shape={mus.shape}, labels shape={labels.shape}")

    plot_pca(mus, labels,  save_path=f"{args.save_dir}/latent_pca.png")
    plot_umap(mus, labels, save_path=f"{args.save_dir}/latent_umap.png")
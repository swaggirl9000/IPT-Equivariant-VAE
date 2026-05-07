"""
interpolate.py

Latent space interpolation between two ModelNet10 point clouds.

Instead of gradient-based inversion (which is unreliable), this script:
  1. Encodes two shapes to latent means mu_a and mu_b
  2. Interpolates linearly between them
  3. Decodes each interpolated mu to Zernike coefficients
  4. Plots the coefficient norm heatmap across the interpolation path
  5. Plots the two endpoint input shapes as 3D scatter plots

Usage
-----
python interpolate.py \
    --checkpoint checkpoint_modelnet10_lmax10_R8_leb29.pt \
    --l_max 10 --R 8 --lebedev 29 \
    --idx_a 0 --idx_b 50 \
    --steps 8 \
    --save_dir ./interpolations
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from get_directions import get_directions
from get_modelnet import PointCloudModelNet
from pipeline import IPTVAEPipeline
from train import load_checkpoint


# ── model helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_mu(model: IPTVAEPipeline, pc: torch.Tensor) -> torch.Tensor:
    """Encode a point cloud and return the posterior mean (no sampling)."""
    _, _, _, _, mu, _ = model(pc)
    return mu   # (1, latent_dim)


@torch.no_grad()
def decode_from_mu(model: IPTVAEPipeline, mu: torch.Tensor) -> torch.Tensor:
    """
    Run only the decoder half of EquivariantVAE + EquivariantDecoder.

    Mirrors the decoder block in vae.forward():
        mu → dec_lin1 → dec_lin2 → dec_lin3 → dec_out → v_raw
    Then:
        v_raw → model.decoder → c_pred
    """
    vae   = model.vae
    h     = vae._gated_layer(vae.dec_lin1, mu)
    h     = vae._gated_layer(vae.dec_lin2, h)
    h     = vae._gated_layer(vae.dec_lin3, h)
    v_raw = vae.dec_out(h)
    return model.decoder(v_raw)   # (1, n_sh, R)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_endpoint_shapes(
    pc_a:    torch.Tensor,
    pc_b:    torch.Tensor,
    label_a: int,
    label_b: int,
    save_path: str,
):
    """3D scatter plots of the two input shapes side by side."""
    CATEGORY_NAMES = {
        0: "bathtub", 1: "bed", 2: "chair", 3: "desk", 4: "dresser",
        5: "monitor", 6: "night_stand", 7: "sofa", 8: "table", 9: "toilet",
    }
    name_a = CATEGORY_NAMES.get(label_a, str(label_a))
    name_b = CATEGORY_NAMES.get(label_b, str(label_b))

    fig = plt.figure(figsize=(8, 4))
    for i, (pc, name) in enumerate([(pc_a, name_a), (pc_b, name_b)]):
        ax  = fig.add_subplot(1, 2, i + 1, projection="3d")
        pts = pc.squeeze(0).cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1, c="steelblue", alpha=0.7)
        ax.set_title(f"Shape {'A' if i == 0 else 'B'}: {name}", fontsize=10)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle("Interpolation endpoints", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved endpoint shapes → {save_path}")


def plot_coefficient_heatmap(
    all_norms: np.ndarray,
    labels:    list,
    save_path: str,
):
    """
    Heatmap of Zernike coefficient norms per spherical harmonic degree l
    across interpolation steps.

    all_norms : (n_steps, l_max+1)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        all_norms.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xlabel("Interpolation step", fontsize=11)
    ax.set_ylabel("Spherical harmonic degree  l", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(all_norms.shape[1]))
    plt.colorbar(im, ax=ax, label="coefficient norm")
    plt.title("Zernike coefficient norms across latent interpolation", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved coefficient heatmap → {save_path}")


def plot_coefficient_lines(
    all_norms: np.ndarray,
    labels:    list,
    save_path: str,
):
    """
    Line plot — one curve per l — showing how each frequency band
    changes across the interpolation path.
    """
    l_max  = all_norms.shape[1] - 1
    steps  = np.arange(len(labels))
    cmap   = matplotlib.colormaps["plasma"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for l in range(l_max + 1):
        color = cmap(l / max(l_max, 1))
        ax.plot(steps, all_norms[:, l], color=color,
                linewidth=1.5, label=f"l={l}")

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Interpolation step", fontsize=11)
    ax.set_ylabel("Coefficient norm", fontsize=11)
    ax.set_title("Per-degree Zernike norms across latent interpolation", fontsize=12)
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved coefficient line plot → {save_path}")


def plot_latent_pca(
    mu_list:   list,
    labels:    list,
    save_path: str,
):
    """
    Project interpolated latent vectors onto their first two PCA components
    and plot the path through latent space.
    """
    vecs = torch.cat(mu_list, dim=0).cpu().numpy()   # (n_steps, latent_dim)

    mean  = vecs.mean(axis=0)
    vecs_c = vecs - mean
    _, _, Vt = np.linalg.svd(vecs_c, full_matrices=False)
    proj  = vecs_c @ Vt[:2].T   # (n_steps, 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(proj[:, 0], proj[:, 1], "-o", color="steelblue",
            linewidth=2, markersize=6, zorder=2)

    for i, (x, y) in enumerate(proj):
        ax.annotate(labels[i], (x, y),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.scatter(proj[0,  0], proj[0,  1], s=80, color="green",
               zorder=3, label="Shape A")
    ax.scatter(proj[-1, 0], proj[-1, 1], s=80, color="red",
               zorder=3, label="Shape B")
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("Interpolation path in latent space (PCA)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved latent PCA plot → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def interpolate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # load model
    state_dict, ckpt_config = load_checkpoint(args.checkpoint, device)
    l_max         = ckpt_config.get("l_max",        args.l_max)
    R             = ckpt_config.get("R",             args.R)
    lebedev_order = ckpt_config.get("lebedev_order", args.lebedev)

    dirs, weights = get_directions(lebedev_order)
    dirs, weights = dirs.to(device), weights.to(device)

    model = IPTVAEPipeline(dirs, weights, l_max=l_max, R=R).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[INFO] Loaded: {args.checkpoint}  |  "
          f"l_max={l_max}  R={R}  lebedev={lebedev_order}")

    # load dataset
    dataset = PointCloudModelNet(
        root       = args.data_root,
        num_points = args.n_points,
        split      = "test",
        categories = 10,
    )
    print(f"[INFO] Dataset: {len(dataset)} test shapes")

    os.makedirs(args.save_dir, exist_ok=True)

    # pick two shapes
    pc_a, label_a = dataset[args.idx_a]
    pc_b, label_b = dataset[args.idx_b]
    pc_a = pc_a.unsqueeze(0).to(device)
    pc_b = pc_b.unsqueeze(0).to(device)
    print(f"[INFO] Shape A: idx={args.idx_a}  label={label_a}")
    print(f"[INFO] Shape B: idx={args.idx_b}  label={label_b}")

    # encode to latent means
    mu_a = encode_mu(model, pc_a)
    mu_b = encode_mu(model, pc_b)
    print(f"[INFO] Latent dim: {mu_a.shape[-1]}")

    # build interpolation
    alphas  = torch.linspace(0.0, 1.0, args.steps, device=device)
    mu_list = [(1.0 - a) * mu_a + a * mu_b for a in alphas]
    labels  = [f"α={a.item():.2f}" for a in alphas]

    # decode each interpolated mu and collect coefficient norms
    print("\n[INFO] Decoding interpolated latent vectors...")
    all_norms = []
    for step, (mu_interp, label) in enumerate(zip(mu_list, labels)):
        c_pred = decode_from_mu(model, mu_interp)   # (1, n_sh, R)

        norms  = []
        sh_idx = 0
        for l in range(l_max + 1):
            m = 2 * l + 1
            block = c_pred[:, sh_idx:sh_idx + m, :]
            norms.append(block.norm().item())
            sh_idx += m
        all_norms.append(norms)

        # save raw coefficients
        np.save(
            os.path.join(args.save_dir, f"step_{step:02d}_alpha{alphas[step].item():.2f}_coeffs.npy"),
            c_pred.squeeze(0).cpu().numpy(),
        )
        print(f"  Step {step+1}/{args.steps}  {label}  "
              f"coeff norm={c_pred.norm().item():.4f}")

    all_norms = np.array(all_norms)   # (n_steps, l_max+1)
    np.save(os.path.join(args.save_dir, "all_norms.npy"), all_norms)

    # save input point clouds
    np.save(os.path.join(args.save_dir, "shape_a.npy"),
            pc_a.squeeze(0).cpu().numpy())
    np.save(os.path.join(args.save_dir, "shape_b.npy"),
            pc_b.squeeze(0).cpu().numpy())

    # ── plots ────────────────────────────────────────────────────────────────
    print("\n[INFO] Generating plots...")

    plot_endpoint_shapes(
        pc_a, pc_b, int(label_a), int(label_b),
        save_path=os.path.join(args.save_dir, "endpoint_shapes.png"),
    )

    plot_coefficient_heatmap(
        all_norms, labels,
        save_path=os.path.join(args.save_dir, "interp_heatmap.png"),
    )

    plot_coefficient_lines(
        all_norms, labels,
        save_path=os.path.join(args.save_dir, "interp_lines.png"),
    )

    plot_latent_pca(
        mu_list, labels,
        save_path=os.path.join(args.save_dir, "latent_pca.png"),
    )

    print(f"\n[INFO] Done. All files saved to: {args.save_dir}")
    print("  endpoint_shapes.png  — the two input point clouds")
    print("  interp_heatmap.png   — coefficient norms per l across interpolation")
    print("  interp_lines.png     — same data as line plot per degree")
    print("  latent_pca.png       — path through latent space (PCA projection)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,   required=True)
    parser.add_argument("--l_max",      type=int,   required=True)
    parser.add_argument("--R",          type=int,   required=True)
    parser.add_argument("--lebedev",    type=int,   default=29)
    parser.add_argument("--data_root",  type=str,
                        default="/home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10")
    parser.add_argument("--idx_a",      type=int,   default=0)
    parser.add_argument("--idx_b",      type=int,   default=50)
    parser.add_argument("--steps",      type=int,   default=8)
    parser.add_argument("--n_points",   type=int,   default=1024)
    parser.add_argument("--save_dir",   type=str,   default="./interpolations")
    args = parser.parse_args()
    interpolate(args)

# import argparse
# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from get_directions import get_directions
# from get_modelnet import PointCloudModelNet
# from get_zernikegrams import compute_pointwise_coefficients
# from pipeline import IPTVAEPipeline
# from train import load_checkpoint


# # ── helpers ───────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def encode_mu(model: IPTVAEPipeline, pc: torch.Tensor) -> torch.Tensor:
#     """Encode a point cloud and return the posterior mean (no sampling)."""
#     _, _, _, _, mu, _ = model(pc)
#     return mu   # (1, latent_dim)


# @torch.no_grad()
# def decode_from_mu(model: IPTVAEPipeline, mu: torch.Tensor) -> torch.Tensor:
#     """
#     Run only the decoder half of the VAE + EquivariantDecoder.

#     Mirrors the decoder block in EquivariantVAE.forward():
#         mu → dec_lin1 → dec_lin2 → dec_lin3 → dec_out  →  v_raw
#                                              → dec_recon →  c_recon (unused here)
#     Then:
#         v_raw → model.decoder → c_pred
#     """
#     vae = model.vae

#     # decoder layers (identical to vae.forward decoder block)
#     h     = vae._gated_layer(vae.dec_lin1, mu)
#     h     = vae._gated_layer(vae.dec_lin2, h)
#     h     = vae._gated_layer(vae.dec_lin3, h)
#     v_raw = vae.dec_out(h)

#     # equivariant decoder → Zernike coefficients
#     c_pred = model.decoder(v_raw)   # (1, n_sh, R)
#     return c_pred


# def invert_zernike_to_pc(
#     c_target: torch.Tensor,
#     l_max:    int,
#     R:        int,
#     n_points: int   = 1024,
#     n_iters:  int   = 300,
#     lr:       float = 0.01,
#     device:   str   = "cpu",
# ) -> torch.Tensor:
#     """Gradient-based inversion: optimise a point cloud to match c_target."""
#     if c_target.dim() == 2:
#         c_target = c_target.unsqueeze(0)
#     c_target = c_target.to(device).detach()

#     pc = torch.randn(1, n_points, 3, device=device)
#     pc = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9
#     pc.requires_grad_(True)

#     optimizer = torch.optim.Adam([pc], lr=lr)

#     for i in range(n_iters):
#         optimizer.zero_grad()
#         norms = pc.norm(dim=-1, keepdim=True)
#         pc_bounded = torch.where(norms > 1.0, pc / norms.clamp(min=1e-8), pc)
        
#         c_rec = compute_pointwise_coefficients(pc_bounded, l_max=l_max, R=R)

#         loss   = torch.tensor(0.0, device=device)
#         sh_idx = 0
#         for l in range(l_max + 1):
#             m  = 2 * l + 1
#             a  = c_rec[:, sh_idx:sh_idx + m, :].reshape(1, -1)
#             b  = c_target[:, sh_idx:sh_idx + m, :].reshape(1, -1)
#             loss = loss + (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
#             sh_idx += m
#         loss = loss / (l_max + 1)

#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(f"    [invert] iter {i+1}/{n_iters}  loss={loss.item():.5f}")

#     with torch.no_grad():
#         norms = pc.norm(dim=-1, keepdim=True)
#         pc_rec = torch.where(norms > 1.0, pc / norms.clamp(min=1e-8), pc)
        
#     return pc_rec.detach()


# def plot_interpolation(point_clouds: list, labels: list, save_path: str):
#     """Save a row of 3D scatter plots."""
#     n   = len(point_clouds)
#     fig = plt.figure(figsize=(3 * n, 3))

#     for i, (pc, label) in enumerate(zip(point_clouds, labels)):
#         ax = fig.add_subplot(1, n, i + 1, projection="3d")
#         pts = pc.squeeze(0).cpu().numpy()
#         ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="steelblue", alpha=0.6)
#         ax.set_title(label, fontsize=7)
#         ax.set_axis_off()
#         ax.set_box_aspect([1, 1, 1])

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"[INFO] Saved figure → {save_path}")


# # ── main ──────────────────────────────────────────────────────────────────────

# def interpolate(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Device: {device}")

#     # load model
#     state_dict, ckpt_config = load_checkpoint(args.checkpoint, device)
#     l_max         = ckpt_config.get("l_max",        args.l_max)
#     R             = ckpt_config.get("R",             args.R)
#     lebedev_order = ckpt_config.get("lebedev_order", args.lebedev)

#     dirs, weights = get_directions(lebedev_order)
#     dirs, weights = dirs.to(device), weights.to(device)

#     model = IPTVAEPipeline(dirs, weights, l_max=l_max, R=R).to(device)
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     print(f"[INFO] Loaded: {args.checkpoint}  |  l_max={l_max}  R={R}  lebedev={lebedev_order}")

#     # load dataset
#     dataset = PointCloudModelNet(
#         root       = args.data_root,
#         num_points = args.n_points,
#         split      = "test",
#         categories = 10,
#     )
#     print(f"[INFO] Dataset: {len(dataset)} test shapes")

#     os.makedirs(args.save_dir, exist_ok=True)

#     # pick two shapes
#     pc_a, label_a = dataset[args.idx_a]
#     pc_b, label_b = dataset[args.idx_b]
#     pc_a = pc_a.unsqueeze(0).to(device)
#     pc_b = pc_b.unsqueeze(0).to(device)
#     print(f"[INFO] Shape A: idx={args.idx_a}  label={label_a}")
#     print(f"[INFO] Shape B: idx={args.idx_b}  label={label_b}")

#     # encode to latent means
#     mu_a = encode_mu(model, pc_a)   # (1, latent_dim)
#     mu_b = encode_mu(model, pc_b)
#     print(f"[INFO] Latent dim: {mu_a.shape[-1]}")

#     # save source and target point clouds
#     np.save(os.path.join(args.save_dir, "shape_a.npy"), pc_a.squeeze(0).cpu().numpy())
#     np.save(os.path.join(args.save_dir, "shape_b.npy"), pc_b.squeeze(0).cpu().numpy())

#     # interpolate, decode, invert
#     alphas       = torch.linspace(0.0, 1.0, args.steps, device=device)
#     point_clouds = []
#     step_labels  = []

#     for step, alpha in enumerate(alphas):
#         alpha_val = alpha.item()
#         mu_interp = (1.0 - alpha) * mu_a + alpha * mu_b

#         c_interp = decode_from_mu(model, mu_interp)   # (1, n_sh, R)

#         print(f"\n[Step {step+1}/{args.steps}]  alpha={alpha_val:.2f} — inverting...")
#         pc_interp = invert_zernike_to_pc(
#             c_target = c_interp,
#             l_max    = l_max,
#             R        = R,
#             n_points = args.n_points,
#             n_iters  = args.n_iters,
#             lr       = args.lr,
#             device   = str(device),
#         )

#         point_clouds.append(pc_interp)
#         step_labels.append(f"α={alpha_val:.2f}")

#         np.save(
#             os.path.join(args.save_dir, f"step_{step:02d}_alpha{alpha_val:.2f}.npy"),
#             pc_interp.squeeze(0).cpu().numpy(),
#         )

#     # plot all steps in one figure
#     plot_interpolation(
#         point_clouds,
#         step_labels,
#         save_path=os.path.join(args.save_dir, "interpolation.png"),
#     )

#     print(f"\n[INFO] Done. All files saved to: {args.save_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint", type=str,   required=True)
#     parser.add_argument("--l_max",      type=int,   required=True)
#     parser.add_argument("--R",          type=int,   required=True)
#     parser.add_argument("--lebedev",    type=int,   default=29)
#     parser.add_argument("--data_root",  type=str,
#                         default="/home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10")
#     parser.add_argument("--idx_a",      type=int,   default=0,
#                         help="Dataset index of shape A")
#     parser.add_argument("--idx_b",      type=int,   default=50,
#                         help="Dataset index of shape B")
#     parser.add_argument("--steps",      type=int,   default=8,
#                         help="Number of interpolation steps including endpoints")
#     parser.add_argument("--n_points",   type=int,   default=1024)
#     parser.add_argument("--n_iters",    type=int,   default=300,
#                         help="Gradient descent steps for Zernike inversion")
#     parser.add_argument("--lr",         type=float, default=0.01)
#     parser.add_argument("--save_dir",   type=str,   default="./interpolations")
#     args = parser.parse_args()
#     interpolate(args)
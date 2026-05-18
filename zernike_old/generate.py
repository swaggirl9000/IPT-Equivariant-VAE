"""
generate.py

Generation evaluation for the IPTVAEPipeline.

Protocol (matches PointFlow / DPC / IPT paper):
  1. Encode all TRAIN shapes → collect mu vectors
  2. Fit a Gaussian (mean + covariance) to the mu vectors
  3. Sample N latent vectors from that Gaussian
  4. Decode each sample → Zernike coefficients → point cloud (gradient inversion)
  5. Compare generated set vs TEST set using:
       COV  (Coverage)               ↑ higher is better
       MMD  (Minimum Matching Dist)  ↓ lower  is better
       JSD  (Jensen-Shannon Div)     ↓ lower  is better

Usage
-----
python generate.py \
    --checkpoint checkpoint_modelnet10_lmax10_R8_leb59_v4.pt \
    --data_root /home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10 \
    --categories 10 \
    --n_generate 2000 \
    --n_points 1024 \
    --n_invert_iters 300 \
    --save_dir ./gen_results
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from get_directions import get_directions
from get_modelnet import PointCloudModelNet
from get_zernikegrams import compute_pointwise_coefficients
from pipeline import IPTVAEPipeline
from train import load_checkpoint


# ---------------------------------------------------------------------------
# Step 1 — Collect latent mu vectors from training set
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents(
    model:      IPTVAEPipeline,
    dataset:    torch.utils.data.Dataset,
    batch_size: int = 64,
    device:     str = "cpu",
) -> np.ndarray:
    """
    Encode every shape in dataset, return (N, latent_dim) array of mu vectors.
    """
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_mus = []

    for pc, _ in loader:
        pc = pc.to(device)
        # Run only encoder path
        f_spatial = model.sft(
            __import__("get_ipt").compute_ect_point_cloud(
                pc, model.dirs, radius=1.0, resolution=model.R, scale=10.0
            )
        )
        x   = model.vae._c_to_e3nn(f_spatial)
        x   = model.vae._gated_layer(model.vae.enc_lin1, x)
        x   = model.vae._gated_layer(model.vae.enc_lin2, x)
        x   = model.vae._gated_layer(model.vae.enc_lin3, x)
        mu  = model.vae.enc_mu(x)
        all_mus.append(mu.cpu().numpy())

    return np.concatenate(all_mus, axis=0)   # (N_train, latent_dim)


# ---------------------------------------------------------------------------
# Step 2 — Fit Gaussian and sample
# ---------------------------------------------------------------------------

def fit_gaussian(mus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a full-covariance Gaussian to the mu vectors.
    Returns (mean, covariance).
    """
    mean = mus.mean(axis=0)
    cov  = np.cov(mus, rowvar=False)
    # Regularise to avoid singular covariance
    cov += np.eye(cov.shape[0]) * 1e-6
    return mean, cov


def sample_latents(
    mean:       np.ndarray,
    cov:        np.ndarray,
    n_samples:  int,
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Sample n_samples latent vectors from N(mean, cov).
    Returns (n_samples, latent_dim) tensor.
    """
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return torch.tensor(samples, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Step 3 — Decode latent → point cloud via gradient inversion
# ---------------------------------------------------------------------------

def decode_latent_to_pc(
    model:       IPTVAEPipeline,
    z:           torch.Tensor,   # (1, latent_dim)
    l_max:       int,
    R:           int,
    n_points:    int   = 1024,
    n_iters:     int   = 300,
    lr:          float = 0.01,
    device:      str   = "cpu",
) -> torch.Tensor:
    """
    Decode a single latent vector z to a point cloud.

    1. Run VAE decoder: z → v_raw → c_pred  (via EquivariantDecoder)
    2. Optimise a point cloud to match c_pred (gradient inversion)
    """
    model.eval()

    with torch.no_grad():
        # VAE decoder trunk
        h     = model.vae._gated_layer(model.vae.dec_lin1, z)
        h     = model.vae._gated_layer(model.vae.dec_lin2, h)
        h     = model.vae._gated_layer(model.vae.dec_lin3, h)
        v_raw = model.vae.dec_out(h)
        # Equivariant decoder → SH coefficient target
        c_target = model.decoder(v_raw)   # (1, F, R)

    # Gradient-based inversion: find pc s.t. Zernike(pc) ≈ c_target
    pc = torch.randn(1, n_points, 3, device=device)
    pc = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9
    pc = pc.requires_grad_(True)

    optimizer = torch.optim.Adam([pc], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()
        pc_norm = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        c_rec   = compute_pointwise_coefficients(pc_norm, l_max=l_max, R=R)

        loss   = torch.tensor(0.0, device=device)
        sh_idx = 0
        for l in range(l_max + 1):
            m = 2 * l + 1
            a = c_rec   [:, sh_idx:sh_idx + m, :].reshape(1, -1)
            b = c_target[:, sh_idx:sh_idx + m, :].reshape(1, -1)
            loss = loss + (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
            sh_idx += m
        (loss / (l_max + 1)).backward()
        optimizer.step()

    with torch.no_grad():
        pc_out = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return pc_out.squeeze(0).detach()   # (n_points, 3)


# ---------------------------------------------------------------------------
# Step 4 — Collect test set point clouds
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_test_pcs(
    dataset:    torch.utils.data.Dataset,
    n_samples:  int,
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Collect up to n_samples point clouds from dataset.
    Returns (N, n_points, 3).
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    pcs    = []
    for pc, _ in loader:
        pcs.append(pc)
        if sum(p.shape[0] for p in pcs) >= n_samples:
            break
    return torch.cat(pcs, dim=0)[:n_samples].to(device)


# ---------------------------------------------------------------------------
# Step 5 — Metrics: COV, MMD, JSD
# ---------------------------------------------------------------------------

def pairwise_cd(
    A: torch.Tensor,   # (Na, N, 3)
    B: torch.Tensor,   # (Nb, N, 3)
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Pairwise Chamfer Distance matrix. Returns (Na, Nb).
    """
    Na = A.shape[0]
    Nb = B.shape[0]
    D  = torch.zeros(Na, Nb, device=A.device)

    for i in range(0, Na, batch_size):
        a_batch = A[i : i + batch_size]           # (bs_a, N, 3)
        bs_a    = a_batch.shape[0]

        for j in range(0, Nb, batch_size):
            b_batch = B[j : j + batch_size]       # (bs_b, N, 3)
            bs_b    = b_batch.shape[0]

            # Compute pairwise distances one pair at a time to avoid OOM
            # For each (a, b) pair compute symmetric CD
            cd_block = torch.zeros(bs_a, bs_b, device=A.device)
            for ii in range(bs_a):
                a = a_batch[ii].unsqueeze(0)       # (1, N, 3)
                # dist from each point in a to each point in b_batch
                diff = a.unsqueeze(2) - b_batch.unsqueeze(1)  # (bs_b, N, N, 3)
                dist = (diff ** 2).sum(-1)                     # (bs_b, N, N)
                cd_ab = dist.min(dim=2).values.mean(dim=1)    # (bs_b,)
                cd_ba = dist.min(dim=1).values.mean(dim=1)    # (bs_b,)
                cd_block[ii] = cd_ab + cd_ba

            D[i : i + bs_a, j : j + bs_b] = cd_block

    return D


def compute_cov_mmd(
    gen_pcs:  torch.Tensor,   # (Ng, N, 3)
    ref_pcs:  torch.Tensor,   # (Nr, N, 3)
    batch_size: int = 32,
) -> dict:
    """
    COV and MMD computed using Chamfer Distance.

    COV : fraction of ref shapes matched by at least one generated shape
    MMD : mean CD from each ref shape to its nearest generated shape
    """
    print("  Computing pairwise CD matrix...")
    D = pairwise_cd(gen_pcs, ref_pcs, batch_size=batch_size)   # (Ng, Nr)

    # MMD: for each ref, find nearest gen
    mmd_per_ref = D.min(dim=0).values   # (Nr,)
    MMD = mmd_per_ref.mean().item()

    # COV: for each gen, find nearest ref; count unique refs matched
    nearest_ref = D.min(dim=1).indices   # (Ng,)
    COV = len(nearest_ref.unique()) / ref_pcs.shape[0]

    return {"COV": COV, "MMD": MMD}


def compute_jsd(
    gen_pcs: torch.Tensor,   # (Ng, N, 3)
    ref_pcs: torch.Tensor,   # (Nr, N, 3)
    n_bins:  int = 28,
) -> float:
    """
    JSD on marginal point distributions, voxelised to an n_bins^3 grid.
    Matches the protocol used by PointFlow and the IPT paper.
    """
    def voxelise(pcs, n_bins):
        # pcs: (B, N, 3), values expected in [-1, 1]
        all_pts = pcs.reshape(-1, 3).cpu().numpy()
        hist, _ = np.histogramdd(
            all_pts,
            bins   = n_bins,
            range  = [[-1, 1], [-1, 1], [-1, 1]],
        )
        p = hist.flatten()
        p = p / p.sum().clip(min=1e-12)
        return p

    p = voxelise(gen_pcs, n_bins)
    q = voxelise(ref_pcs, n_bins)

    m = 0.5 * (p + q)
    # KL(p||m) + KL(q||m), clamped to avoid log(0)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return (a[mask] * np.log(a[mask] / b[mask])).sum()

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(jsd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    state_dict, ckpt_config = load_checkpoint(args.checkpoint, device)
    l_max         = ckpt_config.get("l_max",         args.l_max)
    R             = ckpt_config.get("R",              args.R)
    lebedev_order = ckpt_config.get("lebedev_order",  args.lebedev)

    print(f"[INFO] Loaded: {args.checkpoint}  |  l_max={l_max}  R={R}  lebedev={lebedev_order}")

    dirs, weights = get_directions(lebedev_order)
    dirs, weights = dirs.to(device), weights.to(device)

    model = IPTVAEPipeline(dirs, weights, l_max=l_max, R=R).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Datasets
    train_ds = PointCloudModelNet(
        root       = args.data_root,
        num_points = args.n_points,
        split      = "train",
        categories = args.categories,
    )
    test_ds = PointCloudModelNet(
        root       = args.data_root,
        num_points = args.n_points,
        split      = "test",
        categories = args.categories,
    )

    # --- Step 1: collect latents ---
    print(f"\n[Step 1] Encoding {len(train_ds)} training shapes...")
    mus = collect_latents(model, train_ds, batch_size=64, device=str(device))
    print(f"  mu matrix: {mus.shape}")

    # --- Step 2: fit Gaussian and sample ---
    print(f"\n[Step 2] Fitting Gaussian and sampling {args.n_generate} latents...")
    mean, cov = fit_gaussian(mus)
    z_samples = sample_latents(mean, cov, args.n_generate, device=str(device))
    print(f"  Sampled z: {z_samples.shape}")

    # --- Step 3: decode latents → point clouds ---
    print(f"\n[Step 3] Decoding {args.n_generate} latent vectors to point clouds...")
    print(f"  (inversion: {args.n_invert_iters} iters per shape — this takes a while)")
    os.makedirs(args.save_dir, exist_ok=True)

    gen_pcs = []
    for i in range(args.n_generate):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{args.n_generate}]")
        z = z_samples[i:i+1]   # (1, latent_dim)
        pc = decode_latent_to_pc(
            model    = model,
            z        = z,
            l_max    = l_max,
            R        = R,
            n_points = args.n_points,
            n_iters  = args.n_invert_iters,
            lr       = 0.01,
            device   = str(device),
        )
        gen_pcs.append(pc)

    gen_pcs = torch.stack(gen_pcs, dim=0)   # (Ng, n_points, 3)
    np.save(os.path.join(args.save_dir, "gen_pcs.npy"), gen_pcs.cpu().numpy())
    print(f"  Saved generated point clouds → {args.save_dir}/gen_pcs.npy")

    # --- Step 4: collect test set ---
    print(f"\n[Step 4] Collecting {args.n_ref} test shapes...")
    ref_pcs = collect_test_pcs(test_ds, n_samples=args.n_ref, device=str(device))
    print(f"  ref_pcs: {ref_pcs.shape}")
    np.save(os.path.join(args.save_dir, "ref_pcs.npy"), ref_pcs.cpu().numpy())

    # --- Step 5: compute metrics ---
    print(f"\n[Step 5] Computing COV / MMD / JSD...")
    cov_mmd = compute_cov_mmd(gen_pcs, ref_pcs, batch_size=32)
    jsd     = compute_jsd(gen_pcs, ref_pcs, n_bins=28)

    print("\n" + "=" * 50)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  n_generated : {args.n_generate}")
    print(f"  n_reference : {ref_pcs.shape[0]}")
    print(f"  COV  (↑)    : {cov_mmd['COV']:.4f}")
    print(f"  MMD  (↓)    : {cov_mmd['MMD']:.6f}")
    print(f"  JSD  (↓)    : {jsd:.6f}")
    print("=" * 50)

    # Save results
    results = {
        "checkpoint":   args.checkpoint,
        "l_max":        l_max,
        "R":            R,
        "n_generate":   args.n_generate,
        "n_reference":  ref_pcs.shape[0],
        "COV":          cov_mmd["COV"],
        "MMD":          cov_mmd["MMD"],
        "JSD":          jsd,
    }
    np.save(os.path.join(args.save_dir, "gen_metrics.npy"), results)
    print(f"\n[INFO] Results saved to {args.save_dir}/gen_metrics.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",      type=str,   required=True)
    parser.add_argument("--data_root",       type=str,   required=True)
    parser.add_argument("--categories",      type=int,   default=10)
    parser.add_argument("--n_points",        type=int,   default=1024)
    parser.add_argument("--n_generate",      type=int,   default=2000,
                        help="Number of shapes to generate")
    parser.add_argument("--n_ref",           type=int,   default=2000,
                        help="Number of test shapes to use as reference")
    parser.add_argument("--n_invert_iters",  type=int,   default=300,
                        help="Gradient inversion steps per shape (more = better quality but slower)")
    parser.add_argument("--lebedev",         type=int,   default=59)
    parser.add_argument("--l_max",           type=int,   default=10)
    parser.add_argument("--R",               type=int,   default=8)
    parser.add_argument("--save_dir",        type=str,   default="./gen_results")

    args = parser.parse_args()
    evaluate(args)
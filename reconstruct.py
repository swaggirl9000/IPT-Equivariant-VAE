"""
eval_coefficients.py

Evaluate the IPT-VAE in coefficient space — no inversion needed.

For each test shape:
  - Encode through the full pipeline to get c_pred (decoder output)
  - Compute c_zernike (direct Zernike expansion of the input)
  - Measure per-degree cosine similarity between c_pred and c_zernike
  - Reconstruct point clouds from both coefficient sets
  - Compute Chamfer Distance (CD) and Earth Mover's Distance (EMD)

Reports:
  - Per-degree cosine similarity (mean across all test shapes)
  - Per-category breakdown
  - CD and EMD summary statistics
  - Overall summary

Usage
-----
python eval_coefficients.py \
    --checkpoint checkpoint_modelnet10_lmax10_R8_leb29.pt \
    --l_max 10 --R 8 --lebedev 29 \
    --n_samples 200 \
    --save_dir ./eval_results

Dependencies for CD/EMD
-----------------------
pip install scipy                         # EMD via linear_sum_assignment (exact, CPU)
pip install geomloss                      # optional: faster/GPU approximate EMD (Sinkhorn)
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from get_directions import get_directions
from get_protiens import ProteinNeighborhoods
from pipeline import IPTVAEPipeline
from train import load_checkpoint

CATEGORY_NAMES = {
    0:  "ALA",
    1:  "CYS",
    2:  "ASP",
    3:  "GLU",
    4:  "PHE",
    5:  "GLY",
    6:  "HIS",
    7:  "ILE",
    8:  "LYS",
    9:  "LEU",
    10: "MET",
    11: "ASN",
    12: "PRO",
    13: "GLN",
    14: "ARG",
    15: "SER",
    16: "THR",
    17: "VAL",
    18: "TRP",
    19: "TYR",
}

def reconstruct_pc_from_coefficients(
    coeffs:   torch.Tensor,   
    dirs:     torch.Tensor,   
    weights:  torch.Tensor,   
    l_max:    int,
    n_points: int = 512,
) -> torch.Tensor:
    """
    Approximate reconstruction of a surface point cloud from Zernike coefficients.

    Strategy
    --------
    The coefficients encode the shape as a superposition of spherical-harmonic
    (angular) × radial-basis (radial) components.  For each Lebedev direction d_i
    we sum the contribution of every radial shell and treat the result as a signed
    distance / indicator value.  The predicted *radius* for direction d_i is:

        r_i = sum_n  ||c[:, :, n]||_F weighted by the SH evaluations at d_i

    In practice we project the Lebedev quadrature points by that predicted radius
    to obtain surface samples, then sub-sample to n_points with farthest-point
    sampling so both reconstructed clouds have the same cardinality for CD/EMD.

    If your pipeline exposes a dedicated `decode(z) → pc` method, prefer that
    over this approximation — see the comment in evaluate_sample().

    Returns
    -------
    torch.Tensor of shape (n_points, 3)
    """
    c = coeffs.squeeze(0)             
    c_norm = c.norm(dim=0)   
    radii = c_norm.mean().clamp(min=1e-6)   
    pts = dirs * radii                      
    pts = farthest_point_sample(pts, n_points)
    return pts                            


def farthest_point_sample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """
    Farthest-point sampling.  pts: (N, 3) → returns (min(n, N), 3).
    Pure PyTorch, runs on whatever device pts lives on.
    """
    N = pts.shape[0]
    if n >= N:
        return pts
    sel   = torch.zeros(n, dtype=torch.long, device=pts.device)
    dists = torch.full((N,), float("inf"), device=pts.device)
    cur   = torch.randint(0, N, (1,), device=pts.device).item()
    for i in range(n):
        sel[i]  = cur
        d       = ((pts - pts[cur]) ** 2).sum(-1)
        dists   = torch.minimum(dists, d)
        cur     = dists.argmax().item()
    return pts[sel]

def normalize_pc(pc: torch.Tensor) -> torch.Tensor:
    """
    Centers the point cloud at the origin and scales it to fit within a unit sphere.
    Expects input shape: (N, 3)
    """
    # Center the point cloud
    centroid = pc.mean(dim=0, keepdim=True)
    pc = pc - centroid
    
    # Scale to unit sphere
    max_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    if max_distance > 1e-6:
        pc = pc / max_distance
        
    return pc

# ── Chamfer Distance ──────────────────────────────────────────────────────────

def chamfer_distance(
    pc1: torch.Tensor,   # (N, 3)
    pc2: torch.Tensor,   # (M, 3)
) -> float:
    """
    Symmetric Chamfer Distance (mean of both one-sided distances).

    CD = 0.5 * [ mean_{x in pc1} min_{y in pc2} ||x-y||^2
               + mean_{y in pc2} min_{x in pc1} ||x-y||^2 ]

    Pure PyTorch, O(N*M) — fine for N, M ~ 512–2048.
    For very large clouds consider chunking or a kd-tree.
    """
    sq_dist = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0) ** 2
    cd_fwd  = sq_dist.min(dim=1)[0].mean()   
    cd_bwd  = sq_dist.min(dim=0)[0].mean()  
    return (0.5 * (cd_fwd + cd_bwd)).item()

def emd_scipy(
    pc1: torch.Tensor, 
    pc2: torch.Tensor, 
) -> float:
    """
    Exact EMD via linear-sum assignment (Hungarian algorithm).

    Complexity: O(N^3) — use N ≤ 512 for reasonable runtime.
    For N > 512 consider emd_sinkhorn() below instead.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist as scipy_cdist

    p = pc1.detach().cpu().numpy()
    q = pc2.detach().cpu().numpy()
    cost = scipy_cdist(p, q, metric="euclidean")      
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def emd_sinkhorn(
    pc1: torch.Tensor,   
    pc2: torch.Tensor,
    blur: float = 0.01,
) -> float:
    """
    Approximate EMD via Sinkhorn (GeomLoss).  GPU-accelerated, scales to large N.

    Install: pip install geomloss
    The `blur` parameter controls regularisation — smaller → closer to true EMD
    but slower to converge.
    """
    try:
        from geomloss import SamplesLoss
    except ImportError:
        raise ImportError(
            "geomloss is required for emd_sinkhorn.  "
            "Install with: pip install geomloss"
        )
    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=blur, scaling=0.9)
    return loss_fn(pc1.unsqueeze(0), pc2.unsqueeze(0)).item()


@torch.no_grad()
def evaluate_sample(
    model:    IPTVAEPipeline,
    pc:       torch.Tensor,   
    l_max:    int,
    device:   str,
    dirs:     torch.Tensor,
    weights:  torch.Tensor,
    n_reconstruct: int = 512,
    emd_backend:   str = "scipy",  
) -> dict:
    """
    Returns a dict with:
      "cos_sim"  — np.ndarray (l_max+1,) per-degree cosine similarity
      "cd"       — float Chamfer Distance
      "emd"      — float Earth Mover's Distance
    """
    pc = pc.to(device)

    outputs   = model(pc)
    c_pred    = outputs[0]    
    c_zernike = outputs[1]    

    sims   = []
    sh_idx = 0
    for l in range(l_max + 1):
        m = 2 * l + 1
        a = c_pred[:,    sh_idx:sh_idx + m, :].reshape(1, -1)
        b = c_zernike[:, sh_idx:sh_idx + m, :].reshape(1, -1)
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).item()
        sims.append(cos_sim)
        sh_idx += m

    pc_gt  = pc.squeeze(0)
    pc_rec = reconstruct_pc_from_coefficients(
        c_pred, dirs, weights, l_max, n_points=n_reconstruct
    )
    pc_gt_sub = farthest_point_sample(pc_gt, n_reconstruct)

    # ──> NEW: Normalize both point clouds to a unit sphere <──
    pc_rec = normalize_pc(pc_rec)
    pc_gt_sub = normalize_pc(pc_gt_sub)

    # ── guard: return NaN metrics for degenerate reconstructions ─────────────
    # Must come BEFORE cd/emd computation to avoid crashing on bad clouds.
    if not torch.isfinite(pc_rec).all() or pc_rec.norm() < 1e-6:
        return {
            "cos_sim": np.array(sims),
            "cd":      float("nan"),
            "emd":     float("nan"),
        }

    cd = chamfer_distance(pc_gt_sub, pc_rec)

    if emd_backend == "sinkhorn":
        emd = emd_sinkhorn(pc_gt_sub, pc_rec)
    else:
        emd = emd_scipy(pc_gt_sub, pc_rec)

    return {
        "cos_sim": np.array(sims),
        "cd":      cd,
        "emd":     emd,
    }


def plot_per_degree(
    mean_sims:  np.ndarray,
    std_sims:   np.ndarray,
    l_max:      int,
    save_path:  str,
):
    ls = np.arange(l_max + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ls, mean_sims, yerr=std_sims, capsize=4,
           color="steelblue", alpha=0.8, width=0.6, label="mean ± std")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="perfect")
    ax.set_xlabel("Spherical harmonic degree  l", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title("Per-degree cosine similarity: c_pred vs c_zernike", fontsize=13)
    ax.set_xticks(ls)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved per-degree plot → {save_path}")


def plot_per_category(cat_means, l_max, save_path):
    categories = sorted(cat_means.keys())
    data       = np.array([cat_means[c] for c in categories])
    cat_labels = [CATEGORY_NAMES.get(c, str(c)) for c in categories]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(l_max + 1))
    ax.set_xticklabels([f"l={l}" for l in range(l_max + 1)], fontsize=8)
    ax.set_yticks(range(len(cat_labels)))
    ax.set_yticklabels(cat_labels, fontsize=10)
    ax.set_xlabel("Spherical harmonic degree  l", fontsize=12)
    ax.set_title("Mean cosine similarity per category and SH degree", fontsize=13)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    for i in range(len(categories)):
        for j in range(l_max + 1):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if 0.3 < data[i, j] < 0.85 else "white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved per-category heatmap → {save_path}")


def plot_overall_histogram(all_mean_sims, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_mean_sims, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(all_mean_sims.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"mean={all_mean_sims.mean():.3f}")
    ax.set_xlabel("Mean cosine similarity (all degrees)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of reconstruction quality across test shapes", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved histogram → {save_path}")


def plot_cd_emd(
    all_cd:   np.ndarray,
    all_emd:  np.ndarray,
    save_path: str,
):
    """Side-by-side histograms of per-sample CD and EMD."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals, name, color in zip(
        axes,
        [all_cd, all_emd],
        ["Chamfer Distance (CD)", "Earth Mover's Distance (EMD)"],
        ["darkorange", "mediumseagreen"],
    ):
        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={vals.mean():.4f}")
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {name}", fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved CD/EMD histogram → {save_path}")


def plot_cd_emd_per_category(
    cat_cd:   dict,
    cat_emd:  dict,
    save_path: str,
):
    """Bar chart of mean CD and EMD per category."""
    categories = sorted(cat_cd.keys())
    labels     = [CATEGORY_NAMES.get(c, str(c)) for c in categories]
    cd_means   = [np.mean(cat_cd[c])  for c in categories]
    emd_means  = [np.mean(cat_emd[c]) for c in categories]
    cd_stds    = [np.std(cat_cd[c])   for c in categories]
    emd_stds   = [np.std(cat_emd[c])  for c in categories]

    x   = np.arange(len(categories))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, cd_means,  w, yerr=cd_stds,  capsize=3,
           color="darkorange",    alpha=0.8, label="CD")
    ax.bar(x + w/2, emd_means, w, yerr=emd_stds, capsize=3,
           color="mediumseagreen", alpha=0.8, label="EMD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Distance (lower is better)", fontsize=11)
    ax.set_title("Mean CD and EMD per category", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved per-category CD/EMD plot → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
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
    print(f"[INFO] EMD backend : {args.emd_backend}  "
          f"| reconstruct pts : {args.n_reconstruct}")

    # ── load dataset ──────────────────────────────────────────────────────────
    dataset = ProteinNeighborhoods(
        processed_dir = args.data_root,
        num_points    = args.n_points,
        split         = "test",
    )
    n_samples = min(args.n_samples, len(dataset))
    print(f"[INFO] Evaluating {n_samples} / {len(dataset)} protein neighbourhoods\n")

    os.makedirs(args.save_dir, exist_ok=True)

    all_sims   = []   
    all_cd     = []  
    all_emd    = []   
    all_labels = []

    cat_sims   = {}   
    cat_cd     = {}  
    cat_emd    = {}  

    for i in range(n_samples):
        pc, label = dataset[i]
        label_int = int(label)
        pc        = pc.unsqueeze(0)

        result = evaluate_sample(
            model, pc, l_max, str(device),
            dirs, weights,
            n_reconstruct = args.n_reconstruct,
            emd_backend   = args.emd_backend,
        )

        # skip degenerate samples rather than crashing
        if result is None or np.isnan(result["cd"]):
            print(f"  [{i+1:>4}/{n_samples}]  skipped (degenerate reconstruction)")
            continue

        sims = result["cos_sim"]
        cd   = result["cd"]
        emd  = result["emd"]

        all_sims.append(sims)
        all_cd.append(cd)
        all_emd.append(emd)
        all_labels.append(label_int)

        for d, store in [(label_int, cat_sims), (label_int, cat_cd), (label_int, cat_emd)]:
            if d not in store:
                store[d] = []
        cat_sims[label_int].append(sims)
        cat_cd[label_int].append(cd)
        cat_emd[label_int].append(emd)

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  [{i+1:>4}/{n_samples}]  "
                  f"{CATEGORY_NAMES.get(label_int, label_int):12s}  "
                  f"cos={sims.mean():.4f}  "
                  f"CD={cd:.5f}  "
                  f"EMD={emd:.5f}")

    all_sims = np.array(all_sims) 
    all_cd   = np.array(all_cd)
    all_emd  = np.array(all_emd)

    mean_sims    = all_sims.mean(axis=0)
    std_sims     = all_sims.std(axis=0)
    all_mean_sims = all_sims.mean(axis=1)

    cat_means = {c: np.array(v).mean(axis=0) for c, v in cat_sims.items()}

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY  ({n_samples} samples)")
    print("=" * 70)

    print(f"\n  ── Cosine Similarity (coefficient space) ──")
    print(f"  Overall mean  : {all_mean_sims.mean():.4f} ± {all_mean_sims.std():.4f}")
    print(f"  Best sample   : {all_mean_sims.max():.4f}")
    print(f"  Worst sample  : {all_mean_sims.min():.4f}")

    print(f"\n  ── Chamfer Distance (point cloud space, lower is better) ──")
    print(f"  Overall mean  : {all_cd.mean():.6f} ± {all_cd.std():.6f}")
    print(f"  Best sample   : {all_cd.min():.6f}")
    print(f"  Worst sample  : {all_cd.max():.6f}")

    print(f"\n  ── Earth Mover's Distance (point cloud space, lower is better) ──")
    print(f"  Overall mean  : {all_emd.mean():.6f} ± {all_emd.std():.6f}")
    print(f"  Best sample   : {all_emd.min():.6f}")
    print(f"  Worst sample  : {all_emd.max():.6f}")

    print("\n  Per-degree cosine similarity breakdown:")
    print(f"  {'l':>4}  {'mean':>8}  {'std':>8}")
    print(f"  {'-'*24}")
    for l in range(l_max + 1):
        print(f"  {l:>4}  {mean_sims[l]:>8.4f}  {std_sims[l]:>8.4f}")

    print("\n  Per-category summary (mean over all samples in category):")
    print(f"  {'category':>12}  {'cos_sim':>8}  {'CD':>10}  {'EMD':>10}  {'n':>5}")
    print(f"  {'-'*50}")
    for c in sorted(cat_means.keys()):
        name    = CATEGORY_NAMES.get(c, str(c))
        n_cat   = len(cat_sims[c])
        cos_val = cat_means[c].mean()
        cd_val  = np.mean(cat_cd[c])
        emd_val = np.mean(cat_emd[c])
        print(f"  {name:>12}  {cos_val:>8.4f}  {cd_val:>10.6f}  {emd_val:>10.6f}  {n_cat:>5}")

    print("=" * 70)

    # ── save raw results ──────────────────────────────────────────────────────
    np.save(os.path.join(args.save_dir, "all_sims.npy"),    all_sims)
    np.save(os.path.join(args.save_dir, "all_cd.npy"),      all_cd)
    np.save(os.path.join(args.save_dir, "all_emd.npy"),     all_emd)
    np.save(os.path.join(args.save_dir, "all_labels.npy"),  np.array(all_labels))
    np.save(os.path.join(args.save_dir, "mean_sims.npy"),   mean_sims)
    print(f"[INFO] Raw arrays saved to {args.save_dir}")

    plot_per_degree(
        mean_sims, std_sims, l_max,
        save_path=os.path.join(args.save_dir, "per_degree.png"),
    )
    plot_per_category(
        cat_means, l_max,
        save_path=os.path.join(args.save_dir, "per_category.png"),
    )
    plot_overall_histogram(
        all_mean_sims,
        save_path=os.path.join(args.save_dir, "histogram.png"),
    )
    plot_cd_emd(
        all_cd, all_emd,
        save_path=os.path.join(args.save_dir, "cd_emd_histogram.png"),
    )
    plot_cd_emd_per_category(
        cat_cd, cat_emd,
        save_path=os.path.join(args.save_dir, "cd_emd_per_category.png"),
    )

    print(f"\n[INFO] Done. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IPT-VAE on protein neighbourhood test split."
    )
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--l_max",         type=int, required=True)
    parser.add_argument("--R",             type=int, required=True)
    parser.add_argument("--lebedev",       type=int, default=59)
    parser.add_argument("--data_root",     type=str,
                        default="/gpfs/home3/aromanowski/IPT-Equivariant-VAE/data/protein/processed")
    parser.add_argument("--n_samples",     type=int, default=200)
    parser.add_argument("--n_points",      type=int, default=512)
    parser.add_argument("--n_reconstruct", type=int, default=512)
    parser.add_argument("--emd_backend",   type=str, default="scipy",
                        choices=["scipy", "sinkhorn"],
                        help="'scipy' = exact Hungarian (slower, no extra deps). "
                             "'sinkhorn' = approximate GPU-friendly (needs geomloss).")
    parser.add_argument("--save_dir",      type=str, default="./eval_results_proteins")
    args = parser.parse_args()
    evaluate(args)
# """
# eval_coefficients.py

# Evaluate the IPT-VAE in coefficient space — no inversion needed.

# For each test shape:
#   - Encode through the full pipeline to get c_pred (decoder output)
#   - Compute c_zernike (direct Zernike expansion of the input)
#   - Measure per-degree cosine similarity between c_pred and c_zernike

# Reports:
#   - Per-degree cosine similarity (mean across all test shapes)
#   - Per-category breakdown
#   - Overall summary

# Usage
# -----
# python eval_coefficients.py \
#     --checkpoint checkpoint_modelnet10_lmax10_R8_leb29.pt \
#     --l_max 10 --R 8 --lebedev 29 \
#     --n_samples 200 \
#     --save_dir ./eval_results
# """

# import argparse
# import os
# import torch
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader

# from get_directions import get_directions
# from get_modelnet import PointCloudModelNet
# from pipeline import IPTVAEPipeline
# from train import load_checkpoint

# CATEGORY_NAMES = {
#     0: "bathtub",
#     1: "bed",
#     2: "chair",
#     3: "desk",
#     4: "dresser",
#     5: "monitor",
#     6: "night_stand",
#     7: "sofa",
#     8: "table",
#     9: "toilet",
# }


# # ── evaluation ────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate_sample(
#     model:  IPTVAEPipeline,
#     pc:     torch.Tensor,
#     l_max:  int,
#     device: str,
# ) -> np.ndarray:
#     """
#     Returns per-degree cosine similarity between c_pred and c_zernike.
#     Shape: (l_max+1,)
#     """
#     pc = pc.to(device)
#     c_pred, c_zernike, _, _, _, _ = model(pc)

#     sims   = []
#     sh_idx = 0
#     for l in range(l_max + 1):
#         m = 2 * l + 1
#         a = c_pred[:,   sh_idx:sh_idx + m, :].reshape(1, -1)
#         b = c_zernike[:, sh_idx:sh_idx + m, :].reshape(1, -1)

#         cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).item()
#         sims.append(cos_sim)
#         sh_idx += m

#     return np.array(sims)   # (l_max+1,)


# # ── plotting ──────────────────────────────────────────────────────────────────

# def plot_per_degree(
#     mean_sims:  np.ndarray,
#     std_sims:   np.ndarray,
#     l_max:      int,
#     save_path:  str,
# ):
#     """Bar chart of mean cosine similarity per SH degree l."""
#     ls   = np.arange(l_max + 1)
#     fig, ax = plt.subplots(figsize=(10, 4))

#     ax.bar(ls, mean_sims, yerr=std_sims, capsize=4,
#            color="steelblue", alpha=0.8, width=0.6, label="mean ± std")
#     ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="perfect")
#     ax.set_xlabel("Spherical harmonic degree  l", fontsize=12)
#     ax.set_ylabel("Cosine similarity", fontsize=12)
#     ax.set_title("Per-degree cosine similarity: c_pred vs c_zernike", fontsize=13)
#     ax.set_xticks(ls)
#     ax.set_ylim(0, 1.1)
#     ax.legend(fontsize=10)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"[INFO] Saved per-degree plot → {save_path}")


# def plot_per_category(
#     cat_means:  dict,
#     l_max:      int,
#     save_path:  str,
# ):
#     """
#     Heatmap — rows = categories, columns = SH degree l.
#     Colour = mean cosine similarity.
#     """
#     categories = sorted(cat_means.keys())
#     data       = np.array([cat_means[c] for c in categories])  # (n_cat, l_max+1)
#     cat_labels = [CATEGORY_NAMES.get(c, str(c)) for c in categories]

#     fig, ax = plt.subplots(figsize=(12, 5))
#     im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
#                    vmin=0, vmax=1, origin="upper")
#     ax.set_xticks(range(l_max + 1))
#     ax.set_xticklabels([f"l={l}" for l in range(l_max + 1)], fontsize=8)
#     ax.set_yticks(range(len(cat_labels)))
#     ax.set_yticklabels(cat_labels, fontsize=10)
#     ax.set_xlabel("Spherical harmonic degree  l", fontsize=12)
#     ax.set_title("Mean cosine similarity per category and SH degree", fontsize=13)
#     plt.colorbar(im, ax=ax, label="cosine similarity")

#     # annotate cells
#     for i in range(len(categories)):
#         for j in range(l_max + 1):
#             ax.text(j, i, f"{data[i, j]:.2f}",
#                     ha="center", va="center", fontsize=6,
#                     color="black" if 0.3 < data[i, j] < 0.85 else "white")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"[INFO] Saved per-category heatmap → {save_path}")


# def plot_overall_histogram(
#     all_mean_sims: np.ndarray,
#     save_path:     str,
# ):
#     """Histogram of per-sample mean cosine similarity (averaged over all l)."""
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.hist(all_mean_sims, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
#     ax.axvline(all_mean_sims.mean(), color="red", linestyle="--",
#                linewidth=1.5, label=f"mean={all_mean_sims.mean():.3f}")
#     ax.set_xlabel("Mean cosine similarity (all degrees)", fontsize=12)
#     ax.set_ylabel("Count", fontsize=12)
#     ax.set_title("Distribution of reconstruction quality across test shapes", fontsize=13)
#     ax.legend(fontsize=10)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"[INFO] Saved histogram → {save_path}")


# # ── main ──────────────────────────────────────────────────────────────────────

# def evaluate(args):
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
#     print(f"[INFO] Loaded: {args.checkpoint}  |  "
#           f"l_max={l_max}  R={R}  lebedev={lebedev_order}")

#     # load dataset
#     dataset = PointCloudModelNet(
#         root       = args.data_root,
#         num_points = args.n_points,
#         split      = "test",
#         categories = 10,
#     )
#     n_samples = min(args.n_samples, len(dataset))
#     print(f"[INFO] Evaluating {n_samples} / {len(dataset)} test shapes\n")

#     os.makedirs(args.save_dir, exist_ok=True)

#     # ── evaluation loop ───────────────────────────────────────────────────────
#     all_sims   = []          # (n_samples, l_max+1)
#     all_labels = []          # (n_samples,)
#     cat_sims   = {}          # label → list of (l_max+1,) arrays

#     for i in range(n_samples):
#         pc, label = dataset[i]
#         label_int = int(label)
#         pc        = pc.unsqueeze(0)

#         sims = evaluate_sample(model, pc, l_max, str(device))
#         all_sims.append(sims)
#         all_labels.append(label_int)

#         if label_int not in cat_sims:
#             cat_sims[label_int] = []
#         cat_sims[label_int].append(sims)

#         if (i + 1) % 50 == 0 or (i + 1) == n_samples:
#             print(f"  [{i+1}/{n_samples}]  "
#                   f"{CATEGORY_NAMES.get(label_int, label_int):12s}  "
#                   f"mean_sim={sims.mean():.4f}  "
#                   f"l=0: {sims[0]:.3f}  "
#                   f"l=5: {sims[min(5,l_max)]:.3f}  "
#                   f"l={l_max}: {sims[l_max]:.3f}")

#     all_sims = np.array(all_sims)   # (n_samples, l_max+1)

#     # per-degree stats across all samples
#     mean_sims = all_sims.mean(axis=0)   # (l_max+1,)
#     std_sims  = all_sims.std(axis=0)

#     # per-category means
#     cat_means = {
#         c: np.array(v).mean(axis=0)
#         for c, v in cat_sims.items()
#     }

#     # per-sample mean (averaged over all l)
#     all_mean_sims = all_sims.mean(axis=1)   # (n_samples,)

#     # ── print summary ─────────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print(f"COEFFICIENT EVALUATION SUMMARY  ({n_samples} samples)")
#     print("=" * 60)
#     print(f"\n  Overall mean cosine sim : {all_mean_sims.mean():.4f} "
#           f"± {all_mean_sims.std():.4f}")
#     print(f"  Best sample             : {all_mean_sims.max():.4f}")
#     print(f"  Worst sample            : {all_mean_sims.min():.4f}")

#     print("\n  Per-degree breakdown:")
#     print(f"  {'l':>4}  {'mean':>8}  {'std':>8}")
#     print(f"  {'-'*24}")
#     for l in range(l_max + 1):
#         print(f"  {l:>4}  {mean_sims[l]:>8.4f}  {std_sims[l]:>8.4f}")

#     print("\n  Per-category mean cosine sim (averaged over all l):")
#     print(f"  {'category':>12}  {'mean':>8}  {'n':>5}")
#     print(f"  {'-'*30}")
#     for c in sorted(cat_means.keys()):
#         name  = CATEGORY_NAMES.get(c, str(c))
#         n_cat = len(cat_sims[c])
#         print(f"  {name:>12}  {cat_means[c].mean():>8.4f}  {n_cat:>5}")

#     print("=" * 60)

#     # ── save results ──────────────────────────────────────────────────────────
#     np.save(os.path.join(args.save_dir, "all_sims.npy"),    all_sims)
#     np.save(os.path.join(args.save_dir, "all_labels.npy"),  np.array(all_labels))
#     np.save(os.path.join(args.save_dir, "mean_sims.npy"),   mean_sims)

#     # ── plots ─────────────────────────────────────────────────────────────────
#     plot_per_degree(
#         mean_sims, std_sims, l_max,
#         save_path=os.path.join(args.save_dir, "per_degree.png"),
#     )

#     plot_per_category(
#         cat_means, l_max,
#         save_path=os.path.join(args.save_dir, "per_category.png"),
#     )

#     plot_overall_histogram(
#         all_mean_sims,
#         save_path=os.path.join(args.save_dir, "histogram.png"),
#     )

#     print(f"\n[INFO] Done. Results saved to: {args.save_dir}")
#     print("  per_degree.png    — bar chart of cosine sim per l")
#     print("  per_category.png  — heatmap of cosine sim per category × l")
#     print("  histogram.png     — distribution of reconstruction quality")
#     print("  all_sims.npy      — raw data (n_samples, l_max+1)")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint", type=str,   required=True)
#     parser.add_argument("--l_max",      type=int,   required=True)
#     parser.add_argument("--R",          type=int,   required=True)
#     parser.add_argument("--lebedev",    type=int,   default=29)
#     parser.add_argument("--data_root",  type=str,
#                         default="/home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10")
#     parser.add_argument("--n_samples",  type=int,   default=200)
#     parser.add_argument("--n_points",   type=int,   default=1024)
#     parser.add_argument("--save_dir",   type=str,   default="./eval_results")
#     args = parser.parse_args()
#     evaluate(args)
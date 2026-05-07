"""
eval_coefficients.py

Evaluate the IPT-VAE in coefficient space — no inversion needed.

For each test shape:
  - Encode through the full pipeline to get c_pred (decoder output)
  - Compute c_zernike (direct Zernike expansion of the input)
  - Measure per-degree cosine similarity between c_pred and c_zernike

Reports:
  - Per-degree cosine similarity (mean across all test shapes)
  - Per-category breakdown
  - Overall summary

Usage
-----
python eval_coefficients.py \
    --checkpoint checkpoint_modelnet10_lmax10_R8_leb29.pt \
    --l_max 10 --R 8 --lebedev 29 \
    --n_samples 200 \
    --save_dir ./eval_results
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
from get_modelnet import PointCloudModelNet
from pipeline import IPTVAEPipeline
from train import load_checkpoint

CATEGORY_NAMES = {
    0: "bathtub",
    1: "bed",
    2: "chair",
    3: "desk",
    4: "dresser",
    5: "monitor",
    6: "night_stand",
    7: "sofa",
    8: "table",
    9: "toilet",
}


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sample(
    model:  IPTVAEPipeline,
    pc:     torch.Tensor,
    l_max:  int,
    device: str,
) -> np.ndarray:
    """
    Returns per-degree cosine similarity between c_pred and c_zernike.
    Shape: (l_max+1,)
    """
    pc = pc.to(device)
    c_pred, c_zernike, _, _, _, _ = model(pc)

    sims   = []
    sh_idx = 0
    for l in range(l_max + 1):
        m = 2 * l + 1
        a = c_pred[:,   sh_idx:sh_idx + m, :].reshape(1, -1)
        b = c_zernike[:, sh_idx:sh_idx + m, :].reshape(1, -1)

        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).item()
        sims.append(cos_sim)
        sh_idx += m

    return np.array(sims)   # (l_max+1,)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_per_degree(
    mean_sims:  np.ndarray,
    std_sims:   np.ndarray,
    l_max:      int,
    save_path:  str,
):
    """Bar chart of mean cosine similarity per SH degree l."""
    ls   = np.arange(l_max + 1)
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


def plot_per_category(
    cat_means:  dict,
    l_max:      int,
    save_path:  str,
):
    """
    Heatmap — rows = categories, columns = SH degree l.
    Colour = mean cosine similarity.
    """
    categories = sorted(cat_means.keys())
    data       = np.array([cat_means[c] for c in categories])  # (n_cat, l_max+1)
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

    # annotate cells
    for i in range(len(categories)):
        for j in range(l_max + 1):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if 0.3 < data[i, j] < 0.85 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved per-category heatmap → {save_path}")


def plot_overall_histogram(
    all_mean_sims: np.ndarray,
    save_path:     str,
):
    """Histogram of per-sample mean cosine similarity (averaged over all l)."""
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


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
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
    n_samples = min(args.n_samples, len(dataset))
    print(f"[INFO] Evaluating {n_samples} / {len(dataset)} test shapes\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── evaluation loop ───────────────────────────────────────────────────────
    all_sims   = []          # (n_samples, l_max+1)
    all_labels = []          # (n_samples,)
    cat_sims   = {}          # label → list of (l_max+1,) arrays

    for i in range(n_samples):
        pc, label = dataset[i]
        label_int = int(label)
        pc        = pc.unsqueeze(0)

        sims = evaluate_sample(model, pc, l_max, str(device))
        all_sims.append(sims)
        all_labels.append(label_int)

        if label_int not in cat_sims:
            cat_sims[label_int] = []
        cat_sims[label_int].append(sims)

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  [{i+1}/{n_samples}]  "
                  f"{CATEGORY_NAMES.get(label_int, label_int):12s}  "
                  f"mean_sim={sims.mean():.4f}  "
                  f"l=0: {sims[0]:.3f}  "
                  f"l=5: {sims[min(5,l_max)]:.3f}  "
                  f"l={l_max}: {sims[l_max]:.3f}")

    all_sims = np.array(all_sims)   # (n_samples, l_max+1)

    # per-degree stats across all samples
    mean_sims = all_sims.mean(axis=0)   # (l_max+1,)
    std_sims  = all_sims.std(axis=0)

    # per-category means
    cat_means = {
        c: np.array(v).mean(axis=0)
        for c, v in cat_sims.items()
    }

    # per-sample mean (averaged over all l)
    all_mean_sims = all_sims.mean(axis=1)   # (n_samples,)

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"COEFFICIENT EVALUATION SUMMARY  ({n_samples} samples)")
    print("=" * 60)
    print(f"\n  Overall mean cosine sim : {all_mean_sims.mean():.4f} "
          f"± {all_mean_sims.std():.4f}")
    print(f"  Best sample             : {all_mean_sims.max():.4f}")
    print(f"  Worst sample            : {all_mean_sims.min():.4f}")

    print("\n  Per-degree breakdown:")
    print(f"  {'l':>4}  {'mean':>8}  {'std':>8}")
    print(f"  {'-'*24}")
    for l in range(l_max + 1):
        print(f"  {l:>4}  {mean_sims[l]:>8.4f}  {std_sims[l]:>8.4f}")

    print("\n  Per-category mean cosine sim (averaged over all l):")
    print(f"  {'category':>12}  {'mean':>8}  {'n':>5}")
    print(f"  {'-'*30}")
    for c in sorted(cat_means.keys()):
        name  = CATEGORY_NAMES.get(c, str(c))
        n_cat = len(cat_sims[c])
        print(f"  {name:>12}  {cat_means[c].mean():>8.4f}  {n_cat:>5}")

    print("=" * 60)

    # ── save results ──────────────────────────────────────────────────────────
    np.save(os.path.join(args.save_dir, "all_sims.npy"),    all_sims)
    np.save(os.path.join(args.save_dir, "all_labels.npy"),  np.array(all_labels))
    np.save(os.path.join(args.save_dir, "mean_sims.npy"),   mean_sims)

    # ── plots ─────────────────────────────────────────────────────────────────
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

    print(f"\n[INFO] Done. Results saved to: {args.save_dir}")
    print("  per_degree.png    — bar chart of cosine sim per l")
    print("  per_category.png  — heatmap of cosine sim per category × l")
    print("  histogram.png     — distribution of reconstruction quality")
    print("  all_sims.npy      — raw data (n_samples, l_max+1)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,   required=True)
    parser.add_argument("--l_max",      type=int,   required=True)
    parser.add_argument("--R",          type=int,   required=True)
    parser.add_argument("--lebedev",    type=int,   default=29)
    parser.add_argument("--data_root",  type=str,
                        default="/home/aromanowski/IPT-Equivariant-VAE/data/ModelNet10")
    parser.add_argument("--n_samples",  type=int,   default=200)
    parser.add_argument("--n_points",   type=int,   default=1024)
    parser.add_argument("--save_dir",   type=str,   default="./eval_results")
    args = parser.parse_args()
    evaluate(args)
import argparse
import os
import sys

import torch
import torch.nn.functional as F

from loaders import load_config
from metrics.evaluation import EMD_CD
from transforms.ecttransform import Transform, TransformConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_recon_3d(pcs_recon, pcs_gt, num_pc=8, filename="plot.png"):
    recon = pcs_recon.detach().cpu().numpy()
    gt = pcs_gt.detach().cpu().numpy()
    n_plot = min(num_pc, len(recon), len(gt))
    
    fig = plt.figure(figsize=(4 * n_plot, 8))
    for i in range(n_plot):
        ax_gt = fig.add_subplot(2, n_plot, i + 1, projection='3d')
        ax_gt.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2], c='#1f77b4', s=3, alpha=0.6)
        ax_gt.set_title(f"GT {i}")
        ax_gt.axis('off')
        
        ax_rec = fig.add_subplot(2, n_plot, n_plot + i + 1, projection='3d')
        ax_rec.scatter(recon[i, :, 0], recon[i, :, 1], recon[i, :, 2], c='#d62728', s=3, alpha=0.6)
        ax_rec.set_title(f"Recon {i}")
        ax_rec.axis('off')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_config",
        required=True,
        default=None,
        type=str,
        help="Encoder Configuration YAML file",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Run a small subset.",
    )
    args = parser.parse_args()

    dev: bool = args.dev

    (
        dataconfig,
        transformconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    ) = load_config(args.encoder_config)

    results_base_dir = "results"
    if dev:
        results_base_dir += "_dev"
    results_base_dir += f"/{loggerconfig.results_dir}"

    print(f"Loading evaluated tensors from: {results_base_dir}")
    
    pcs_recon = torch.load(f"{results_base_dir}/pcs_recon.pt", weights_only=True)
    pcs_gt    = torch.load(f"{results_base_dir}/pcs_gt.pt",    weights_only=True)

    n_plot = min(8, pcs_recon.shape[0], pcs_gt.shape[0])

    print("Generating 3D reconstruction plots...")
    plot_recon_3d(
        10 * pcs_recon[:n_plot],
        10 * pcs_gt[:n_plot],
        num_pc=n_plot,
        filename=f"{results_base_dir}/pcs_recon_test.png",
    )

    # print("\nCalculating CD and EMD scores...")
    # geometry_results = EMD_CD(pcs_recon, pcs_gt, batch_size=128, accelerated_cd=True)
    # cd, emd = geometry_results
    # print(f"CD:  {cd:.6f}")
    # print(f"EMD: {emd:.6f}")
    print("\nCalculating CD and EMD scores...")
    
    from metrics.evaluation import _pairwise_EMD_CD_
    cd_matrix, emd_matrix = _pairwise_EMD_CD_(pcs_recon, pcs_gt, batch_size=128)
    
    cd_scores = torch.diagonal(cd_matrix)
    emd_scores = torch.diagonal(emd_matrix)
    
    n = len(cd_scores)
    print(f"CD:  {cd_scores.mean():.6f} ± {cd_scores.std() / n**0.5:.6f}")
    print(f"EMD: {emd_scores.mean():.6f} ± {emd_scores.std() / n**0.5:.6f}")
    
    print("\nCalculating Cosine Similarity of IPT features...")

    loss_transform_config = TransformConfig(module="", ectconfig=modelconfig.ectlossconfig)
    losstransform = Transform(loss_transform_config).to(DEVICE)

    cosine_sims = []

    for i in range(0, len(pcs_gt), 128):
        batch_gt    = pcs_gt[i:i + 128].to(DEVICE)
        batch_recon = pcs_recon[i:i + 128].to(DEVICE)
        
        ipt_gt    = losstransform(batch_gt).flatten(start_dim=1)
        ipt_recon = losstransform(batch_recon).flatten(start_dim=1)
        
        sim = F.cosine_similarity(ipt_gt, ipt_recon, dim=1)  # (batch,)
        cosine_sims.append(sim.cpu())

    all_sims = torch.cat(cosine_sims)  # (N,)
    n = len(all_sims)
    print(f"Mean Cosine Similarity: {all_sims.mean():.6f} ± {all_sims.std() / n**0.5:.6f}\n")
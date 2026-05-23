import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lightning.fabric import Fabric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loaders import load_config, load_model
from metrics.evaluation import EMD_CD

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def plot_recon_3d(pcs_recon, pcs_gt, num_pc=8, filename="plot.png"):
    recon  = pcs_recon.detach().cpu().numpy()
    gt     = pcs_gt.detach().cpu().numpy()
    n_plot = min(num_pc, len(recon), len(gt))
    fig    = plt.figure(figsize=(4 * n_plot, 8))
    for i in range(n_plot):
        ax = fig.add_subplot(2, n_plot, i + 1, projection='3d')
        ax.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2], c='#1f77b4', s=3, alpha=0.6)
        ax.set_title(f"GT {i}")
        ax.axis('off')
        ax = fig.add_subplot(2, n_plot, n_plot + i + 1, projection='3d')
        ax.scatter(recon[i, :, 0], recon[i, :, 1], recon[i, :, 2], c='#d62728', s=3, alpha=0.6)
        ax.set_title(f"Recon {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_config",     required=True, type=str)
    parser.add_argument("--encoder_config", required=True, type=str)
    parser.add_argument("--dev",            default=False, action="store_true")
    args = parser.parse_args()

    (_, _, vae_modelconfig, _, vae_loggerconfig) = load_config(args.vae_config)
    (_, _, enc_modelconfig, _, enc_loggerconfig) = load_config(args.encoder_config)

    vae_results_dir = "results"
    if args.dev:
        vae_results_dir += "_dev"
    vae_results_dir += f"/{vae_loggerconfig.results_dir}"

    enc_results_dir = f"results/{enc_loggerconfig.results_dir}"

    print(f"Loading VAE tensors from: {vae_results_dir}")
    recon_sh  = torch.load(f"{vae_results_dir}/recon_sh.pt",  weights_only=True)
    gt_sh     = torch.load(f"{vae_results_dir}/gt_sh.pt",     weights_only=True)
    recon_ect = torch.load(f"{vae_results_dir}/recon_ect.pt", weights_only=True)
    gt_ect    = torch.load(f"{vae_results_dir}/gt_ect.pt",    weights_only=True)

    print(f"Loading equivariant decoder from: {enc_results_dir}/model.ckpt")
    enc_model = load_model(enc_modelconfig)
    fabric = Fabric(accelerator="auto", precision="32-true")
    fabric.load(f"{enc_results_dir}/model.ckpt", {"model": enc_model})
    enc_model = enc_model.to(DEVICE)
    enc_model.eval()

    #  Generate point clouds 
    print("\nRunning equivariant decoder on GT and reconstructed SH features...")
    batch_size     = 32

    pcs_gt = torch.load(f"{vae_results_dir}/gt_pcs.pt", weights_only=True)

    pcs_recon_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(recon_sh), batch_size)):
            pcs_recon_list.append(enc_model(recon_sh[i:i+batch_size].to(DEVICE)).cpu())
    pcs_recon = torch.cat(pcs_recon_list)

    plot_recon_3d(
        pcs_recon[:8], pcs_gt[:8],
        num_pc=8,
        filename=f"{vae_results_dir}/pcs_recon_test.png",
    )

    print("\nCD and EMD scores...")
    cd_scores, emd_scores = [], []
    for i in range(0, len(pcs_gt), batch_size):
        cd, emd = EMD_CD(
            pcs_recon[i:i + batch_size],
            pcs_gt[i:i + batch_size],
            batch_size=batch_size,
            accelerated_cd=True,
        )
        cd_scores.append(cd)
        emd_scores.append(emd)

    cd_scores  = torch.tensor(cd_scores)
    emd_scores = torch.tensor(emd_scores)
    n = len(cd_scores)
    print(f"  CD:  {cd_scores.mean():.6f} ± {cd_scores.std() / n**0.5:.6f}")
    print(f"  EMD: {emd_scores.mean():.6f} ± {emd_scores.std() / n**0.5:.6f}")

    print("\nECT Cosine Similarity...")
    cosine_sims = []
    for i in range(0, len(gt_ect), batch_size):
        b_gt    = gt_ect[i:i + batch_size].flatten(start_dim=1)
        b_recon = recon_ect[i:i + batch_size].flatten(start_dim=1)
        cosine_sims.append(F.cosine_similarity(b_gt, b_recon, dim=1).cpu())

    all_sims = torch.cat(cosine_sims)
    n_sims   = len(all_sims)
    print(f"  Cosine Similarity: {all_sims.mean():.6f} ± {all_sims.std() / n_sims**0.5:.6f}")
# import argparse
# import torch
# import torch.nn.functional as F

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from loaders import load_config

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# def plot_sh_comparison(gt_sh, recon_sh, num=8, filename="sh_comparison.png"):
#     """Plot GT vs reconstructed SH coefficient magnitudes per sample."""
#     n     = min(num, gt_sh.shape[0])
#     gt    = gt_sh[:n].detach().cpu().float()
#     recon = recon_sh[:n].detach().cpu().float()

#     fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
#     for i in range(n):
#         axes[0, i].imshow(gt[i].numpy(),    aspect='auto', cmap='viridis')
#         axes[0, i].set_title(f"GT {i}")
#         axes[0, i].axis('off')
#         axes[1, i].imshow(recon[i].numpy(), aspect='auto', cmap='viridis')
#         axes[1, i].set_title(f"Recon {i}")
#         axes[1, i].axis('off')
#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved SH comparison to: {filename}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vae_config", required=True, type=str)
#     parser.add_argument("--dev",        default=False, action="store_true")
#     args = parser.parse_args()

#     (_, _, _, _, loggerconfig) = load_config(args.vae_config)

#     results_dir = "results"
#     if args.dev:
#         results_dir += "_dev"
#     results_dir += f"/{loggerconfig.results_dir}"

#     print(f"Loading tensors from: {results_dir}")
#     recon_sh  = torch.load(f"{results_dir}/recon_sh.pt",  weights_only=True)
#     gt_sh     = torch.load(f"{results_dir}/gt_sh.pt",     weights_only=True)
#     recon_ect = torch.load(f"{results_dir}/recon_ect.pt", weights_only=True)
#     gt_ect    = torch.load(f"{results_dir}/gt_ect.pt",    weights_only=True)

#     batch_size = 32

#     # ── SH reconstruction metrics ─────────────────────────────────────── #
#     print("\nSH reconstruction metrics...")
#     sh_cosine_sims = []
#     sh_mse_scores  = []

#     for i in range(0, len(gt_sh), batch_size):
#         b_gt    = gt_sh[i:i + batch_size].flatten(start_dim=1)
#         b_recon = recon_sh[i:i + batch_size].flatten(start_dim=1)
#         sh_cosine_sims.append(F.cosine_similarity(b_gt, b_recon, dim=1).cpu())
#         sh_mse_scores.append(
#             F.mse_loss(b_recon, b_gt, reduction='none').mean(dim=1).cpu()
#         )

#     sh_cos = torch.cat(sh_cosine_sims)
#     sh_mse = torch.cat(sh_mse_scores)
#     n      = len(sh_cos)
#     print(f"  Cosine Similarity : {sh_cos.mean():.6f} ± {sh_cos.std() / n**0.5:.6f}")
#     print(f"  MSE               : {sh_mse.mean():.6f} ± {sh_mse.std() / n**0.5:.6f}")

#     # ── ECT reconstruction metrics ─────────────────────────────────────── #
#     print("\nECT reconstruction metrics...")
#     ect_cosine_sims = []
#     ect_mse_scores  = []

#     for i in range(0, len(gt_ect), batch_size):
#         b_gt    = gt_ect[i:i + batch_size].flatten(start_dim=1)
#         b_recon = recon_ect[i:i + batch_size].flatten(start_dim=1)
#         ect_cosine_sims.append(F.cosine_similarity(b_gt, b_recon, dim=1).cpu())
#         ect_mse_scores.append(
#             F.mse_loss(b_recon, b_gt, reduction='none').mean(dim=1).cpu()
#         )

#     ect_cos = torch.cat(ect_cosine_sims)
#     ect_mse = torch.cat(ect_mse_scores)
#     n       = len(ect_cos)
#     print(f"  Cosine Similarity : {ect_cos.mean():.6f} ± {ect_cos.std() / n**0.5:.6f}")
#     print(f"  MSE               : {ect_mse.mean():.6f} ± {ect_mse.std() / n**0.5:.6f}")

#     # ── Visual comparison ─────────────────────────────────────────────── #
#     print("\nGenerating SH comparison plot...")
#     plot_sh_comparison(
#         gt_sh, recon_sh,
#         num=8,
#         filename=f"{results_dir}/sh_recon_test.png",
#     )
# import argparse
# import os
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm

# from loaders import load_config, load_datamodule, load_model
# from metrics.evaluation import EMD_CD
# from spherical_harmonics import SphericalHarmonicProjection, InverseSphericalHarmonicProjection
# from get_directions import get_directions

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# def plot_recon_3d(pcs_recon, pcs_gt, num_pc=8, filename="plot.png"):
#     recon  = pcs_recon.detach().cpu().numpy()
#     gt     = pcs_gt.detach().cpu().numpy()
#     n_plot = min(num_pc, len(recon), len(gt))
#     fig    = plt.figure(figsize=(4 * n_plot, 8))
#     for i in range(n_plot):
#         ax = fig.add_subplot(2, n_plot, i + 1, projection='3d')
#         ax.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2], c='#1f77b4', s=3, alpha=0.6)
#         ax.set_title(f"GT {i}")
#         ax.axis('off')
#         ax = fig.add_subplot(2, n_plot, n_plot + i + 1, projection='3d')
#         ax.scatter(recon[i, :, 0], recon[i, :, 1], recon[i, :, 2], c='#d62728', s=3, alpha=0.6)
#         ax.set_title(f"Recon {i}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close(fig)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vae_config",     required=True,  type=str)
#     parser.add_argument("--encoder_config", required=True,  type=str)
#     parser.add_argument("--dev",            default=False,  action="store_true")
#     args = parser.parse_args()

#     # ── Load configs ─────────────────────────────────────────────────── #
#     (
#         vae_dataconfig, _, vae_modelconfig,
#         vae_trainerconfig, vae_loggerconfig,
#     ) = load_config(args.vae_config)

#     (
#         enc_dataconfig, _, enc_modelconfig,
#         enc_trainerconfig, enc_loggerconfig,
#     ) = load_config(args.encoder_config)

#     vae_results_dir = f"results/{vae_loggerconfig.results_dir}"
#     enc_results_dir = f"results/{enc_loggerconfig.results_dir}"

#     # ── Load saved tensors from VAE training ─────────────────────────── #
#     print(f"Loading VAE tensors from: {vae_results_dir}")
#     recon_sh = torch.load(f"{vae_results_dir}/recon_sh.pt", weights_only=True)
#     gt_sh    = torch.load(f"{vae_results_dir}/gt_sh.pt",    weights_only=True)
#     recon_ect = torch.load(f"{vae_results_dir}/recon_ect.pt", weights_only=True)
#     gt_ect    = torch.load(f"{vae_results_dir}/gt_ect.pt",    weights_only=True)

#     # ── Load equivariant decoder ──────────────────────────────────────── #
#     print(f"Loading equivariant decoder from: {enc_results_dir}")
#     enc_model = load_model(enc_modelconfig)
#     state     = {"model": enc_model}
#     ckpt_path = f"{enc_results_dir}/model.ckpt"
#     from lightning.fabric import Fabric
#     fabric = Fabric(accelerator="gpu", precision=None)
#     fabric.load(ckpt_path, state)
#     enc_model = enc_model.to(DEVICE)
#     enc_model.eval()

#     # ── SH projection (needed to feed decoder) ───────────────────────── #
#     dirs, weights = get_directions(num_points=vae_modelconfig.ectconfig.num_thetas)
#     dirs    = dirs.to(DEVICE)
#     weights = weights.to(DEVICE)

#     sh_transform = SphericalHarmonicProjection(
#         dirs=dirs, weights=weights, l_max=vae_modelconfig.l_max,
#     ).to(DEVICE)

#     # ── Generate point clouds via decoder ────────────────────────────── #
#     print("\nGenerating point clouds from reconstructed SH features...")
#     pcs_recon_list = []
#     pcs_gt_list    = []
#     batch_size     = 32

#     with torch.no_grad():
#         for i in tqdm(range(0, len(recon_sh), batch_size)):
#             sh_batch_recon = recon_sh[i:i + batch_size].to(DEVICE)
#             sh_batch_gt    = gt_sh[i:i + batch_size].to(DEVICE)
#             pcs_recon_list.append(enc_model(sh_batch_recon).cpu())
#             pcs_gt_list.append(enc_model(sh_batch_gt).cpu())

#     pcs_recon = torch.cat(pcs_recon_list)
#     pcs_gt    = torch.cat(pcs_gt_list)

#     torch.save(pcs_recon, f"{vae_results_dir}/pcs_recon.pt")
#     torch.save(pcs_gt,    f"{vae_results_dir}/pcs_gt.pt")

#     # ── Plots ─────────────────────────────────────────────────────────── #
#     print("\nGenerating reconstruction plots...")
#     plot_recon_3d(
#         pcs_recon[:8], pcs_gt[:8],
#         num_pc=8,
#         filename=f"{vae_results_dir}/pcs_recon_test.png",
#     )

#     # ── CD / EMD with standard error ─────────────────────────────────── #
#     print("\nCalculating CD and EMD scores...")
#     cd_scores  = []
#     emd_scores = []
#     for i in range(0, len(pcs_gt), batch_size):
#         b_recon = pcs_recon[i:i + batch_size]
#         b_gt    = pcs_gt[i:i + batch_size]
#         cd, emd = EMD_CD(b_recon, b_gt, batch_size=batch_size, accelerated_cd=True)
#         cd_scores.append(cd)
#         emd_scores.append(emd)

#     cd_scores  = torch.tensor(cd_scores)
#     emd_scores = torch.tensor(emd_scores)
#     n = len(cd_scores)
#     print(f"CD:  {cd_scores.mean():.6f} ± {cd_scores.std() / n**0.5:.6f}")
#     print(f"EMD: {emd_scores.mean():.6f} ± {emd_scores.std() / n**0.5:.6f}")

#     # ── Cosine similarity on ECT features with standard error ─────────── #
#     print("\nCalculating Cosine Similarity of ECT features...")
#     cosine_sims = []
#     for i in range(0, len(gt_ect), batch_size):
#         b_gt    = gt_ect[i:i + batch_size].flatten(start_dim=1).to(DEVICE)
#         b_recon = recon_ect[i:i + batch_size].flatten(start_dim=1).to(DEVICE)
#         sim = F.cosine_similarity(b_gt, b_recon, dim=1)
#         cosine_sims.append(sim.cpu())

#     all_sims = torch.cat(cosine_sims)
#     n_sims   = len(all_sims)
#     print(f"Mean Cosine Similarity: {all_sims.mean():.6f} ± {all_sims.std() / n_sims**0.5:.6f}")
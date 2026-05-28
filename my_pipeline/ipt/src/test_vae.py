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
    parser.add_argument("--vae_config",          required=True,  type=str)
    parser.add_argument("--encoder_config",      required=True,  type=str)
    parser.add_argument("--encoder_checkpoint",  default=None,   type=str,
                        help="Path to encoder checkpoint. Defaults to "
                             "results/<enc_results_dir>/model.ckpt")
    parser.add_argument("--dev",                 default=False,  action="store_true")
    parser.add_argument("--vae_results_dir", default=None, type=str,
                    help="Override directory for recon_sh.pt / gt_pcs.pt etc.")
    args = parser.parse_args()

    (_, _, vae_modelconfig, _, vae_loggerconfig) = load_config(args.vae_config)
    (_, _, enc_modelconfig, _, enc_loggerconfig) = load_config(args.encoder_config)

    vae_results_dir = "results"
    if args.dev:
        vae_results_dir += "_dev"
    vae_results_dir = args.vae_results_dir or (f"results{'_dev' if args.dev else ''}/{vae_loggerconfig.results_dir}")

    enc_results_dir = f"results/{enc_loggerconfig.results_dir}"

    # Allow overriding the encoder checkpoint path so you can evaluate
    # a model saved to a different directory (e.g. results_joint/...)
    enc_ckpt = args.encoder_checkpoint or f"{enc_results_dir}/model.ckpt"

    print(f"Loading VAE tensors from: {vae_results_dir}")
    recon_sh  = torch.load(f"{vae_results_dir}/recon_sh.pt",  weights_only=True)
    gt_sh     = torch.load(f"{vae_results_dir}/gt_sh.pt",     weights_only=True)
    recon_ect = torch.load(f"{vae_results_dir}/recon_ect.pt", weights_only=True)
    gt_ect    = torch.load(f"{vae_results_dir}/gt_ect.pt",    weights_only=True)

    print(f"Loading equivariant decoder from: {enc_ckpt}")
    enc_model = load_model(enc_modelconfig)
    fabric = Fabric(accelerator="auto", precision="32-true")
    fabric.load(enc_ckpt, {"model": enc_model})
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

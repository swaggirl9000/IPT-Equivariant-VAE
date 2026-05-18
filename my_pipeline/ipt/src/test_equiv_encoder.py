import argparse
import os
import torch
import torch.nn.functional as F

from loaders import load_config
from metrics.evaluation import EMD_CD
from plotting import plot_recon_3d
from transforms.ecttransform import Transform, TransformConfig

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
    
    pcs_recon = torch.load(f"{results_base_dir}/pcs_recon.pt")
    pcs_gt = torch.load(f"{results_base_dir}/pcs_gt.pt")

    plot_recon_3d(
        10 * pcs_recon[:8],
        10 * pcs_gt[:8],
        num_pc=8,
        filename=f"{results_base_dir}/pcs_recon_test.png",
    )

    print("\nCalculating CD and EMD scores...")
    geometry_results = EMD_CD(pcs_recon, pcs_gt, batch_size=128, accelerated_cd=True)
    
    for metric_name, value in geometry_results.items():
        print(f"{metric_name}: {value:.6f}")

    print("\nCalculating Cosine Similarity of IPT features...")
    
    loss_transform_config = TransformConfig(module="", ectconfig=modelconfig.ectlossconfig)
    losstransform = Transform(loss_transform_config).to(DEVICE)

    cosine_sims = []
    
    for i in range(0, len(pcs_gt), 128):
        batch_gt = pcs_gt[i : i + 128].to(DEVICE)
        batch_recon = pcs_recon[i : i + 128].to(DEVICE)
        
        ipt_gt = losstransform(batch_gt).flatten(start_dim=1)
        ipt_recon = losstransform(batch_recon).flatten(start_dim=1)
        
        sim = F.cosine_similarity(ipt_gt, ipt_recon, dim=1)
        cosine_sims.append(sim.cpu())
        
    mean_cosine_sim = torch.cat(cosine_sims).mean().item()
    print(f"Mean Cosine Similarity: {mean_cosine_sim:.6f}\n")
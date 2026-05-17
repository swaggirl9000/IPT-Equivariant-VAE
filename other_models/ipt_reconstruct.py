"""
python reconstruct.py \
    --config  configs/vae_protein.yaml \
    --checkpoint path/to/checkpoint.ckpt \
    --n_samples 200 \
    --save_dir  ./eval_results_encoder
"""

import argparse
import os

import numpy as np
import torch
from lightning.fabric import Fabric
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from get_protiens import ProteinNeighborhoods   
from scipy.spatial.distance import cdist as scipy_cdist
from torch_geometric.data import Data

from layers.ect import EctLayer
from loaders import load_config, load_model


def normalize_pc(pc: torch.Tensor) -> torch.Tensor:
    """Centre and scale to unit sphere.  Input: (N, 3)"""
    pc = pc - pc.mean(dim=0, keepdim=True)
    r  = torch.sqrt((pc ** 2).sum(dim=1)).max()
    if r > 1e-6:
        pc = pc / r
    return pc


def farthest_point_sample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """FPS: (N, 3) → (min(n, N), 3)"""
    N = pts.shape[0]
    if n >= N:
        return pts
    sel   = torch.zeros(n, dtype=torch.long, device=pts.device)
    dists = torch.full((N,), float("inf"), device=pts.device)
    cur   = torch.randint(0, N, (1,), device=pts.device).item()
    for i in range(n):
        sel[i] = cur
        d      = ((pts - pts[cur]) ** 2).sum(-1)
        dists  = torch.minimum(dists, d)
        cur    = dists.argmax().item()
    return pts[sel]


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    sq = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0) ** 2
    return (0.5 * (sq.min(1)[0].mean() + sq.min(0)[0].mean())).item()


def emd_scipy(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    p    = pc1.detach().cpu().numpy()
    q    = pc2.detach().cpu().numpy()
    cost = scipy_cdist(p, q, metric="euclidean")
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())

def cosine_similarity_matched(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    p = pc1.detach().cpu().numpy()
    q = pc2.detach().cpu().numpy()
    
    cost = scipy_cdist(p, q, metric="euclidean")
    row, col = linear_sum_assignment(cost)
    
    matched_pc1 = pc1[row]
    matched_pc2 = pc2[col]
    
    cos_sim = F.cosine_similarity(matched_pc1, matched_pc2, dim=1)
    
    return float(cos_sim.mean().item())


@torch.no_grad()
def evaluate_sample(
    model,
    ect_layer,
    pc_gt: torch.Tensor,  
    device: str,
    n_reconstruct: int = 512,
    emd_backend: str = "scipy",
) -> dict:
    pc_gt = pc_gt.to(device)
    b, n, _ = pc_gt.shape

    idx   = torch.arange(b, device=device).repeat_interleave(n)
    batch = Data(x=pc_gt.reshape(-1, 3))
    ect   = ect_layer(batch, idx).unsqueeze(1)

    output = model(ect)
    if isinstance(output, tuple):
        pc_rec = output[0]
    else:
        pc_rec = output                

    pc_rec = pc_rec.squeeze(0)          
    pc_gt  = pc_gt.squeeze(0)         

    if not torch.isfinite(pc_rec).all() or pc_rec.norm() < 1e-6:
        return {"cd": float("nan"), "emd": float("nan"), "cos_sim": float("nan")}

    pc_gt_sub = farthest_point_sample(pc_gt, n_reconstruct)
    pc_rec    = farthest_point_sample(pc_rec, n_reconstruct)

    pc_gt_sub = normalize_pc(pc_gt_sub)
    pc_rec    = normalize_pc(pc_rec)

    cd = chamfer_distance(pc_gt_sub, pc_rec)

    if emd_backend == "sinkhorn":
        try:
            from geomloss import SamplesLoss
            emd = SamplesLoss("sinkhorn", p=1, blur=0.01)(
                pc_gt_sub.unsqueeze(0), pc_rec.unsqueeze(0)
            ).item()
        except ImportError:
            print("[WARN] geomloss not found, falling back to scipy EMD")
            emd = emd_scipy(pc_gt_sub, pc_rec)
    else:
        emd = emd_scipy(pc_gt_sub, pc_rec)

    cos_sim = cosine_similarity_matched(pc_gt_sub, pc_rec)

    return {"cd": cd, "emd": emd, "cos_sim": cos_sim}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    dataconfig, transformconfig, modelconfig, trainerconfig, loggerconfig = \
        load_config(args.config)

    fabric = Fabric(accelerator="gpu", precision=None)
    model  = load_model(modelconfig)

    state = {"model": model}
    fabric.load(args.checkpoint, state)
    model = fabric.setup_module(model)
    model.eval()
    print(f"[INFO] Loaded  : {args.checkpoint}")

    v = torch.randn(3, modelconfig.ectconfig.num_thetas)
    v /= torch.norm(v, dim=0)
    v = v.to(device)
    ect_layer = EctLayer(config=modelconfig.ectconfig, v=v)
    ect_layer = fabric.setup_module(ect_layer)

    dataset   = ProteinNeighborhoods(
        processed_dir = "/home/aromanowski/IPT-Equivariant-VAE/data/protein/processed",
        split      = "test",
        num_points = 1024,
    )
    n_samples = min(args.n_samples, len(dataset))
    print(f"[INFO] Evaluating {n_samples} / {len(dataset)} test samples\n")

    os.makedirs(args.save_dir, exist_ok=True)

    all_cd  = []
    all_emd = []
    all_cos_sim = []
    for i in range(n_samples):
        sample = dataset[i]
        if isinstance(sample, (list, tuple)):
            pc = sample[0]
        else:
            pc = sample.x 

        pc = pc.unsqueeze(0)   

        result = evaluate_sample(
            model, ect_layer, pc,
            device        = str(device),
            n_reconstruct = args.n_reconstruct,
            emd_backend   = args.emd_backend,
        )

        if np.isnan(result["cd"]):
            print(f"  [{i+1:>4}/{n_samples}]  skipped (degenerate)")
            continue

        all_cd.append(result["cd"])
        all_emd.append(result["emd"])
        all_cos_sim.append(result["cos_sim"])

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  [{i+1:>4}/{n_samples}]  "
                  f"CD={result['cd']:.5f}  EMD={result['emd']:.5f}")

    all_cd  = np.array(all_cd)
    all_emd = np.array(all_emd)
    all_cos_sim = np.array(all_cos_sim)

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  ({len(all_cd)} valid samples)")
    print("=" * 60)
    print(f"\n  Chamfer Distance  (lower is better)")
    print(f"    mean : {all_cd.mean():.6f}  ±  {all_cd.std():.6f}")
    print(f"    min  : {all_cd.min():.6f}")
    print(f"    max  : {all_cd.max():.6f}")
    print(f"\n  Earth Mover's Distance  (lower is better)")
    print(f"    mean : {all_emd.mean():.6f}  ±  {all_emd.std():.6f}")
    print(f"    min  : {all_emd.min():.6f}")
    print(f"    max  : {all_emd.max():.6f}")
    print(f"\n  Cosine Similarity  (higher is better)")
    print(f"    mean : {all_cos_sim.mean():.6f}  ±  {all_cos_sim.std():.6f}")
    print(f"    min  : {all_cos_sim.min():.6f}")
    print(f"    max  : {all_cos_sim.max():.6f}")
    print("=" * 60)

    np.save(os.path.join(args.save_dir, "cd.npy"),  all_cd)
    np.save(os.path.join(args.save_dir, "emd.npy"), all_emd)
    np.save(os.path.join(args.save_dir, "cos_sim.npy"), all_cos_sim)
    print(f"\n[INFO] Raw arrays saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IPT encoder — CD and EMD on test split."
    )
    parser.add_argument("--config",        required=True,
                        help="Path to YAML config (same one used for training)")
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to Fabric checkpoint (.ckpt)")
    parser.add_argument("--n_samples",     type=int, default=200)
    parser.add_argument("--n_reconstruct", type=int, default=512,
                        help="Points used for CD/EMD after FPS sub-sampling")
    parser.add_argument("--emd_backend",   default="scipy",
                        choices=["scipy", "sinkhorn"],
                        help="scipy = exact Hungarian; sinkhorn = approx GPU (needs geomloss)")
    parser.add_argument("--save_dir",      default="./eval_results_encoder")
    args = parser.parse_args()
    evaluate(args)
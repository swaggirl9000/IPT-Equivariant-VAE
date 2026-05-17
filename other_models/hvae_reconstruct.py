"""
python hvae_reconstruct.py \\
    --experiment_dir  /path/to/hvae_run1 \\
    --checkpoint_name lowest_rec_loss_model.pt \\
    --data_path       /path/to/zernikegrams_test-lmax=6-r=10.0-rst_normalization=square.hdf5 \\
    --n_samples       200 \\
    --save_dir        ./hvae_eval_results
"""
import os

import argparse
import json
import sys
import hdf5plugin

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist as scipy_cdist
from torch.utils.data import Dataset, DataLoader

from holographic_vae.models import H_VAE
from holographic_vae.so3.functional import put_dict_on_device
from holographic_vae.cg_coefficients import get_w3j_coefficients
from experiments.protein_neighborhoods.src.utils.data import load_data

class HVAEDataset(Dataset):
    """
    Reads a zernikegram HDF5 file directly and slices the flat array
    into a dictionary using the data_irreps string.
    """

    def __init__(self, hdf5_path: str, data_irreps: str):
        super().__init__()
        self.hdf5_path = hdf5_path
        
        from e3nn import o3
        irreps = o3.Irreps(data_irreps)

        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            fields = data.dtype.names
            zg = torch.tensor(data["zernikegram"][:]).float()

            self.zg_keys = []
            self.zg_data = {}
            start = 0
            
            for mul, ir in irreps:
                l = ir.l
                key = l  
                self.zg_keys.append(key)
                
                dim = mul * (2 * l + 1)
                chunk = zg[:, start : start + dim]
                
                self.zg_data[key] = chunk.view(-1, mul, 2 * l + 1)
                start += dim

            self.X_vec  = (torch.tensor(data["X_vec"][:]).float()
                           if "X_vec" in fields else None)
            self.labels = (torch.tensor(data["label"][:]).long()
                           if "label" in fields else None)
            self.ids    = (data["ids"][:].tolist()
                           if "ids" in fields else None)
            self.rots   = (torch.tensor(data["rot"][:]).float()
                           if "rot" in fields else None)

        self.n = zg.shape[0]
        print(f"[INFO] HVAEDataset: {self.n} samples  |  keys: {self.zg_keys}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        X     = {k: self.zg_data[k][idx] for k in self.zg_keys}
        X_vec = self.X_vec[idx]  if self.X_vec  is not None else torch.zeros(1, 3)
        y     = self.labels[idx] if self.labels  is not None else torch.tensor(-1)
        rot   = (self.rots[idx]  if self.rots    is not None else torch.eye(3).flatten())
        sid   = self.ids[idx]    if self.ids     is not None else idx
        return X, X_vec, y, (rot, sid)

def flatten_irreps_dict(X: dict) -> torch.Tensor:
    """Flatten all irrep tensors into (batch, D_total)."""
    parts = [X[k].reshape(X[k].shape[0], -1) for k in sorted(X.keys())]
    return torch.cat(parts, dim=-1)


def irreps_to_feature_rows(X: dict) -> torch.Tensor:
    """
    Build (n_keys, max_feat_dim) feature matrix from an irreps dict (batch=1).
    Pads smaller irrep feature vectors with zeros to match the largest one 
    so they can be safely stacked.
    """
    flat_tensors = [X[k][0].reshape(-1).float() for k in sorted(X.keys())]
    max_len = max(t.size(0) for t in flat_tensors)
    padded_tensors = [
        torch.nn.functional.pad(t, (0, max_len - t.size(0))) 
        for t in flat_tensors
    ]
    return torch.stack(padded_tensors, dim=0)


def chamfer_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    sq = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0) ** 2
    return (0.5 * (sq.min(1)[0].mean() + sq.min(0)[0].mean())).item()


def emd_scipy(A: torch.Tensor, B: torch.Tensor) -> float:
    p, q = A.cpu().numpy(), B.cpu().numpy()
    cost = scipy_cdist(p, q, metric="euclidean")
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def cosine_sim_flat(A: torch.Tensor, B: torch.Tensor) -> float:
    return F.cosine_similarity(A, B, dim=-1).mean().item()


def load_hvae(experiment_dir: str, checkpoint_name: str, device: str):
    hparams_path = os.path.join(experiment_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    print("[INFO] Calling load_data(valid) to recover data_irreps / norm_factor …")
    datasets, data_irreps, norm_factor = load_data(hparams, splits=["valid"])
    print(f"[INFO] data_irreps : {data_irreps}")
    print(f"[INFO] norm_factor : {norm_factor}")

    hparams["model_hparams"]["input_normalizing_constant"] = norm_factor

    w3j_matrices = get_w3j_coefficients()
    for key in w3j_matrices:
        w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        w3j_matrices[key].requires_grad = False

    model = H_VAE(
        data_irreps,
        w3j_matrices,
        hparams["model_hparams"],
        device,
        normalize_input_at_runtime=False,
    ).to(device)

    ckpt_path = os.path.join(experiment_dir, checkpoint_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[INFO] Loaded checkpoint : {ckpt_path}")
    return model, hparams, data_irreps

@torch.no_grad()
def evaluate_sample(model, X, rot, device) -> dict:
    X     = put_dict_on_device(X, device)
    frame = rot.float().view(-1, 3, 3).to(device)
    
    x_vec_flat = flatten_irreps_dict(X)

    x_reconst_loss, kl_divergence, recon_X, (mean, log_var) = model(
        X, x_vec=x_vec_flat, frame=frame
    )

    if recon_X is None:
        return {
            "cd": float("nan"), "emd": float("nan"), "cos_sim": float("nan"),
        }

    cos_sim = cosine_sim_flat(x_vec_flat, flatten_irreps_dict(recon_X))
    inp_feat, rec_feat = irreps_to_feature_rows(X), irreps_to_feature_rows(recon_X)

    return {
        "cd":       chamfer_distance(inp_feat, rec_feat),
        "emd":      emd_scipy(inp_feat, rec_feat),
        "cos_sim":  cos_sim,
    }


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model, _, data_irreps = load_hvae(args.experiment_dir, args.checkpoint_name, device)

    dataset   = HVAEDataset(args.data_path, data_irreps=str(data_irreps))
    loader    = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    n_samples = min(args.n_samples, len(dataset))
    print(f"[INFO] Evaluating {n_samples} / {len(dataset)} samples\n")

    os.makedirs(args.save_dir, exist_ok=True)

    all_cd, all_emd, all_cos_sim = [], [], []

    for i, (X, _, y, (rot, _)) in enumerate(loader): 
        if i >= n_samples:
            break

        result = evaluate_sample(model, X, rot, device)

        if np.isnan(result["cd"]):
            print(f"  [{i+1:>4}/{n_samples}]  skipped (degenerate output)")
            continue

        all_cd.append(result["cd"])
        all_emd.append(result["emd"])
        all_cos_sim.append(result["cos_sim"])

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(
                f"  [{i+1:>4}/{n_samples}]  "
                f"CD={result['cd']:.5f}  EMD={result['emd']:.5f}  "
                f"CosSim={result['cos_sim']:.4f}"
            )

    all_cd, all_emd = np.array(all_cd), np.array(all_emd)
    all_cos_sim     = np.array(all_cos_sim)

    print("\n" + "=" * 65)
    print(f"EVALUATION SUMMARY  ({len(all_cd)} valid samples)")
    print("=" * 65)
    for label, arr, better in [
        ("Chamfer Distance [SH feature space]", all_cd,      "lower"),
        ("Earth Mover's Distance [SH feat.]",   all_emd,     "lower"),
        ("Cosine Similarity [flat SH]",          all_cos_sim, "higher"),
    ]:
        print(f"\n  {label}  ({better} is better)")
        print(f"    mean : {arr.mean():.6f}  ±  {arr.std():.6f}")
        print(f"    min  : {arr.min():.6f}   max : {arr.max():.6f}")
    print("=" * 65)

    for name, arr in [("hvae_cd", all_cd), ("hvae_emd", all_emd),
                      ("hvae_cos_sim", all_cos_sim)]:
        np.save(os.path.join(args.save_dir, f"{name}.npy"), arr)
    print(f"\n[INFO] Arrays saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir",  required=True,
                        help="Dir with hparams.json and the checkpoint")
    parser.add_argument("--checkpoint_name", default="lowest_rec_loss_model.pt")
    parser.add_argument("--data_path",       required=True,
                        help="Full path to the test zernikegram HDF5 file")
    parser.add_argument("--n_samples",       type=int, default=200)
    parser.add_argument("--save_dir",        default="./hvae_eval_results")
    args = parser.parse_args()
    evaluate(args)
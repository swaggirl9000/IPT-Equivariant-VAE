"""
hvae_equivariance_test.py

SO(3) equivariance test for the H-VAE.

python hvae_equivariance_test.py \\
    --experiment_dir  /path/to/hvae_run1 \\
    --checkpoint_name lowest_rec_loss_model.pt \\
    --data_path       /path/to/zernikegrams_test-lmax=6-r=10.0-rst_normalization=square.hdf5 \\
    --N               500 \\
    --n_trials        5 \\
    --label           hvae_run1
"""
import os
import argparse
import json
import sys

import hdf5plugin
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from e3nn import o3

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

def load_hvae(experiment_dir: str, checkpoint_name: str, device: str):
    hparams_path = os.path.join(experiment_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    print("[INFO] Calling load_data(valid) for data_irreps / norm_factor …")
    datasets, data_irreps, norm_factor = load_data(hparams, splits=["valid"])
    print(f"[INFO] data_irreps : {data_irreps}")
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()
    print(f"[INFO] Loaded checkpoint : {ckpt_path}")
    return model, hparams, data_irreps


def rotate_irreps_dict(X: dict, rot_cpu: torch.Tensor, device: str) -> dict:
    """
    Apply SO(3) rotation to every irrep in X using Wigner-D matrices.
    """
    X_rot = {}
    for l in X.keys():  
        irrep_str   = f"1x{l}e"
        D_l         = o3.Irreps(irrep_str).D_from_matrix(rot_cpu).to(device) 
        v           = X[l].to(device)                              
        
        X_rot[l]    = torch.einsum("ij, bcj -> bci", D_l, v)
    return X_rot


def flatten_dict(X: dict) -> torch.Tensor:
    """Flatten irreps dict → (batch, D_total)."""
    return torch.cat([X[k].reshape(X[k].shape[0], -1) for k in sorted(X.keys())], dim=-1)


@torch.no_grad()
def equivariance_error(
    model,
    X: dict,
    N: int = 100,   
    device: str = "cuda",
) -> np.ndarray:
    
    model.eval()
    X     = put_dict_on_device(X, device)

    frame_id = torch.eye(3, device=device).unsqueeze(0)
    
    x_vec_flat = flatten_dict(X)

    _, _, baseline_recon, _ = model(X, x_vec=x_vec_flat, frame=frame_id)
    if baseline_recon is None:
        raise RuntimeError("model() returned None for recon_X")

    baseline_flat = flatten_dict(baseline_recon)    
    baseline_norm = baseline_flat.norm().item()

    errors = []
    for _ in range(N):
        rot_cpu = o3.rand_matrix()                    

        # F(D(g)·X)
        X_rot = rotate_irreps_dict(X, rot_cpu, device)
        
        x_rot_vec_flat = flatten_dict(X_rot)
        
        _, _, recon_of_rotated, _ = model(X_rot, x_vec=x_rot_vec_flat, frame=frame_id)
        f_of_rotated = flatten_dict(recon_of_rotated)   

        # D(g)·F(X) 
        rotated_baseline = rotate_irreps_dict(baseline_recon, rot_cpu, device)
        d_of_baseline    = flatten_dict(rotated_baseline)  

        diff = (f_of_rotated - d_of_baseline).norm().item()
        errors.append(diff / (baseline_norm + 1e-8))

    return np.array(errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir",  required=True,
                        help="Dir with hparams.json and the checkpoint")
    parser.add_argument("--checkpoint_name", default="lowest_rec_loss_model.pt")
    parser.add_argument("--data_path",       required=True,
                        help="Full path to the test zernikegram HDF5 file")
    parser.add_argument("--N",        type=int, default=500,
                        help="Random rotations per trial")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Number of distinct test samples to use")
    parser.add_argument("--label",    type=str, default="hvae",
                        help="Label printed in the summary line")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model, _, data_irreps = load_hvae(args.experiment_dir, args.checkpoint_name, device)

    dataset     = HVAEDataset(args.data_path, data_irreps=str(data_irreps))
    loader      = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)
    loader_iter = iter(loader)

    all_errors = []
    for trial in range(args.n_trials):
        print(f"Trial {trial + 1}/{args.n_trials} …")
        try:
            X, _, y, (rot, _) = next(loader_iter)  
        except StopIteration:
            loader_iter = iter(loader)
            X, _, y, (rot, _) = next(loader_iter)

        errors = equivariance_error(model, X, N=args.N, device=device) 
        all_errors.extend(errors.tolist())

    all_errors = np.array(all_errors)
    print(f"\nSO(3) equivariance error ({args.label}) — {len(all_errors)} rotations:")
    print(f"  Mean : {all_errors.mean():.4f}")
    print(f"  Max  : {all_errors.max():.4f}")
    print(f"  Std  : {all_errors.std():.4f}")
    print("\n(Score ≈ 0 → perfectly equivariant.  Higher = less equivariant.)")
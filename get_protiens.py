import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset

# Map the 3-letter codes back to the correct integers
CATEGORY_MAP = {
    "ALA": 0,  "CYS": 1,  "ASP": 2,  "GLU": 3,  "PHE": 4,
    "GLY": 5,  "HIS": 6,  "ILE": 7,  "LYS": 8,  "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}

class ProteinNeighborhoods(Dataset):
    def __init__(
        self,
        processed_dir: str,
        num_points:    int   = 512,
        radius_cut:    float = 10.0,
        split:         str   = "train",
        **kwargs,
    ):
        self.num_points = num_points

        all_files = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if f.endswith(".npy")
        ])

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {processed_dir}. "
                "Run pub.py first to generate point clouds from PDBs."
            )

        # ── FIX 1: Shuffle the files deterministically ──
        random.seed(42) # Ensures train/test splits are always exactly the same
        random.shuffle(all_files)

        # deterministic 80/10/10 split
        n       = len(all_files)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        if split == "train":
            self.files = all_files[:n_train]
        elif split in ("val", "valid"):
            self.files = all_files[n_train:n_train + n_val]
        elif split == "test":
            self.files = all_files[n_train + n_val:]
        else:
            self.files = all_files

        print(f"[INFO] ProteinNeighborhoods | split={split} | "
              f"{len(self.files)} / {n} clouds from {processed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        pc = np.load(filepath).astype(np.float32)

        # ── guard: drop NaN/inf rows ──────────────────────────────────
        pc = pc[np.isfinite(pc).all(axis=1)]
        if pc.shape[0] < 10:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)

        # secondary resample
        n = pc.shape[0]
        if n >= self.num_points:
            idx_ = np.random.choice(n, self.num_points, replace=False)
        else:
            idx_ = np.random.choice(n, self.num_points, replace=True)
        pc = pc[idx_]

        # ── normalise to unit sphere (centre + scale) ─────────────────
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        scale = np.abs(pc).max()
        if scale > 1e-6:
            pc = pc / scale
            
        # ── FIX 2: Extract the actual label from the filename ──
        filename = os.path.basename(filepath).upper()
        label_int = 0 
        for cat_name, cat_int in CATEGORY_MAP.items():
            if cat_name in filename: # This will now find "ALA", "CYS", etc.
                label_int = cat_int
                break
        return torch.tensor(pc), torch.tensor(label_int)

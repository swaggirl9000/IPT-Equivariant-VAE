import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class ShapeNetPC15kDataset(Dataset):
    def __init__(self, split_dir: str, n_points: int = 2048):
        self.n_points = n_points
        self.files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".npy")
        ])
        assert len(self.files) > 0, f"No .npy files found in {split_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = torch.from_numpy(np.load(self.files[idx])).float()
        perm = torch.randperm(pc.size(0))[:self.n_points]
        return pc[perm]                  


class ShapeNetDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.root        = cfg.data_dir   
        self.batch_size  = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.n_points    = cfg.get("n_points", 2048)

    def setup(self, stage=None):
        self.train_ds = ShapeNetPC15kDataset(os.path.join(self.root, "train"), self.n_points)
        self.val_ds   = ShapeNetPC15kDataset(os.path.join(self.root, "val"),   self.n_points)
        self.test_ds  = ShapeNetPC15kDataset(os.path.join(self.root, "test"),  self.n_points)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)
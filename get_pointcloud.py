# get_shapenet.py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import ShapeNet as ShapeNetPyG

class PointCloudShapeNet(Dataset):
    """
    Wraps torch_geometric ShapeNet to match the IPT paper setup:
      - Categories: Airplane, Chair, Car  (same as IPT paper)
      - 2048 points per cloud             (same as IPT paper)
      - Points normalised to unit sphere  (same as IPT paper)
      - Returns (pc, label) matching PointCloudMNIST API
    """

    CATEGORIES = ["Airplane", "Chair", "Car"]

    def __init__(
        self,
        root:       str  = "./data/shapenet",
        train:      bool = True,
        num_points: int  = 2048,
        categories: list = None,
    ):
        self.num_points = num_points
        self.categories = categories or self.CATEGORIES

        split = "train" if train else "test"

        self.dataset = ShapeNetPyG(
            root       = root,
            categories = self.categories,
            split      = split,
        )

        # Build a simple integer label per category
        self.cat_to_label = {c: i for i, c in enumerate(self.categories)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data  = self.dataset[idx]
        pos   = data.pos                              # (N, 3) float tensor
        cat   = data.category                        # string e.g. "Airplane"

        # --- subsample / upsample to exactly num_points ---
        N = pos.shape[0]
        if N >= self.num_points:
            indices = torch.randperm(N)[:self.num_points]
        else:
            indices = torch.randint(0, N, (self.num_points,))
        pc = pos[indices]                             # (num_points, 3)

        # --- normalise to unit sphere (IPT paper preprocessing) ---
        pc = pc - pc.mean(dim=0, keepdim=True)        # centre
        scale = pc.norm(dim=-1).max().clamp(min=1e-8)
        pc = pc / scale                               # fits in unit ball

        label = self.cat_to_label.get(cat, 0)
        return pc, label
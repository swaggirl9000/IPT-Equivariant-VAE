import torch
from torch.utils.data import Dataset

try:
    import torch_geometric
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, NormalizeScale
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False

CATEGORY_TO_LABEL = {
    "bathtub":     0,
    "bed":         1,
    "chair":       2,
    "desk":        3,
    "dresser":     4,
    "monitor":     5,
    "night_stand": 6,
    "sofa":        7,
    "table":       8,
    "toilet":      9,
}


def random_rotation_matrix() -> torch.Tensor:
    """Uniform random SO(3) rotation via QR decomposition."""
    M = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(M)
    Q = Q * torch.det(Q).sign()   # ensure det = +1
    return Q


class PointCloudModelNet(Dataset):
    def __init__(
        self,
        root:          str  = "./data/ModelNet10",
        num_points:    int  = 1024,
        split:         str  = "train",
        categories:    int  = 10,
        normalise:     bool = True,
        random_rotate: bool = True,    # ← new
    ):
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required.")

        self.num_points    = num_points
        self.normalise     = normalise
        self.split         = split
        # only rotate during training, never during val/test
        self.random_rotate = random_rotate and (split == "train")

        transform = SamplePoints(num_points, include_normals=False)

        self.tg_dataset = ModelNet(
            root      = root,
            name      = str(categories),
            train     = (split == "train"),
            transform = transform,
        )

    def __len__(self) -> int:
        return len(self.tg_dataset)

    def __getitem__(self, idx: int):
        data  = self.tg_dataset[idx]
        pc    = data.pos                   # (num_points, 3)
        label = int(data.y.item())

        if self.normalise:
            pc = pc - pc.mean(dim=0, keepdim=True)
            scale = pc.norm(dim=-1).max().clamp(min=1e-8)
            pc = pc / scale

        if self.random_rotate:
            R  = random_rotation_matrix()
            pc = pc @ R.T

        return pc.float(), label
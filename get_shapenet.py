"""
get_shapenet.py

Dataset loader for ShapeNetCore.v2.PC15k — the exact dataset used by the
IPT paper (Röell & Rieck, NeurIPS 2025).  Drop-in replacement for the
previous torch_geometric-based loader, with an identical (pc, label) API.

Expected directory layout after downloading and unzipping
ShapeNetCore.v2.PC15k.zip:

    root/
    └── ShapeNetCore.v2.PC15k/
        ├── 02691156/          ← synset ID for Airplane
        │   ├── train/
        │   │   ├── 1a04e3eab45ca15dd86060f189eb133.npy
        │   │   └── ...
        │   ├── val/
        │   └── test/
        ├── 03001627/          ← Chair
        ├── 02958343/          ← Car
        └── ...

Each .npy file is a float32 array of shape (15000, 3) — a dense point
cloud from which we subsample num_points points at random every epoch,
giving stochastic data augmentation for free.

Download
--------
https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j
(linked from https://github.com/aidos-lab/inner-product-transforms)

Usage
-----
    from get_shapenet import PointCloudShapeNetPC15k

    train_ds = PointCloudShapeNetPC15k(
        root       = "./data/ShapeNetCore.v2.PC15k",
        categories = ["Airplane", "Chair", "Car"],
        split      = "train",
        num_points = 2048,
    )
    pc, label = train_ds[0]   # pc: (2048, 3),  label: int
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

SYNSET_TO_CATEGORY = {
    "02691156": "Airplane",       
    "02958343": "Car",            
    "03001627": "Chair",         
}

CATEGORY_TO_SYNSET = {v: k for k, v in SYNSET_TO_CATEGORY.items()}

CATEGORY_TO_LABEL = {cat: i for i, cat in enumerate(sorted(SYNSET_TO_CATEGORY.values()))}


class PointCloudShapeNet(Dataset):
    """
    ShapeNetCore.v2.PC15k point cloud dataset.

    Reads pre-computed dense point clouds (.npy, shape 15000×3) and
    subsamples num_points points per shape on every __getitem__ call,
    giving free stochastic augmentation during training.

    Returns (pc, label) where:
      pc    : Tensor (num_points, 3)  — normalised to unit sphere
      label : int                     — integer category index

    Parameters
    ----------
    root : str
        Path to the unzipped ShapeNetCore.v2.PC15k directory,
        i.e. the folder that contains synset-ID sub-folders like 02691156/.
    categories : list[str] or None
        Category names to include, e.g. ["Airplane", "Chair", "Car"].
        None = all available categories under root.
    split : str
        "train", "val", or "test".  Must match sub-folder names in the
        dataset (the PC15k release uses these exact names).
    num_points : int
        Number of points to subsample per shape.  The IPT paper uses 2048.
    normalise : bool
        Centre each point cloud and scale to unit sphere.
        Must be True to match the IPT paper's preprocessing.
    """

    def __init__(
        self,
        root:        str        = "./data/ShapeNetCore.v2.PC15k",
        categories:  list | None = None,
        split:       str        = "train",
        num_points:  int        = 2048,
        normalise:   bool       = True,
        rotate:      bool       = False, #added 
    ):
        super().__init__()
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.root       = root
        self.split      = split
        self.num_points = num_points
        self.normalise  = normalise
        self.rotate = rotate #added 

        if categories is not None:
            synsets = []
            for cat in categories:
                if cat not in CATEGORY_TO_SYNSET:
                    raise ValueError(
                        f"Unknown category '{cat}'. "
                        f"Valid names: {sorted(CATEGORY_TO_SYNSET.keys())}"
                    )
                synsets.append(CATEGORY_TO_SYNSET[cat])
        else:
            # Use every synset folder that exists under root
            synsets = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
                and d in SYNSET_TO_CATEGORY
            ]

        if len(synsets) == 0:
            raise FileNotFoundError(
                f"No valid synset folders found under '{root}'. "
                "Check that the path points to the unzipped "
                "ShapeNetCore.v2.PC15k directory."
            )

        self._samples: list[tuple[str, int]] = []

        for synset in sorted(synsets):
            cat_name  = SYNSET_TO_CATEGORY[synset]
            label     = CATEGORY_TO_LABEL[cat_name]
            split_dir = os.path.join(root, synset, split)

            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Split directory not found: {split_dir}\n"
                    f"Expected layout: root/{{synset_id}}/{{train|val|test}}/*.npy"
                )

            npy_files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))

            if len(npy_files) == 0:
                raise FileNotFoundError(
                    f"No .npy files found in {split_dir}. "
                    "Check that the zip was extracted correctly."
                )

            n_bad = 0
            for fpath in npy_files:
                if os.path.getsize(fpath) == 0:
                    n_bad += 1
                    continue
                try:
                    np.load(fpath, mmap_mode="r")
                    self._samples.append((fpath, label))
                except Exception:
                    n_bad += 1
            if n_bad:
                import warnings
                warnings.warn(
                    f"[get_shapenet] Skipped {n_bad} corrupt/empty .npy file(s) "
                    f"in {split_dir}."
                )

        if len(self._samples) == 0:
            raise RuntimeError(
                f"Dataset is empty for split='{split}' and "
                f"categories={categories}."
            )

        self._categories_present = sorted({
            SYNSET_TO_CATEGORY[s] for s in synsets
            if SYNSET_TO_CATEGORY[s] in CATEGORY_TO_LABEL
        })

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return (
            f"PointCloudShapeNet("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"categories={self._categories_present}, "
            f"num_points={self.num_points})"
        )

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]

        pc_full = np.load(path).astype(np.float32)   

        N = pc_full.shape[0]
        if N >= self.num_points:
            idx_pts = np.random.choice(N, self.num_points, replace=False)
        else:
            idx_pts = np.random.choice(N, self.num_points, replace=True)

        pc = torch.from_numpy(pc_full[idx_pts])       

        if self.normalise:
            pc = pc - pc.mean(dim=0, keepdim=True)
            scale = pc.norm(dim=-1).max().clamp(min=1e-8)
            pc = pc / scale
        
        #added
        if self.rotate:
            rot_matrix = R.random().as_matrix()
            rot_matrix = torch.from_numpy(rot_matrix).to(pc.dtype)
            
            pc = pc @ rot_matrix.T

        return pc, label


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./data/ShapeNetCore.v2.PC15k"

    print(f"Root: {root}\n")

    for split in ("train", "val", "test"):
        try:
            ds = PointCloudShapeNet(
                root       = root,
                categories = ["Airplane", "Chair", "Car"],
                split      = split,
                num_points = 2048,
            )
            pc, label = ds[0]
            print(f"[{split:5s}]  {len(ds):5d} shapes  |  "
                  f"pc={tuple(pc.shape)}  "
                  f"range=[{pc.min():.3f}, {pc.max():.3f}]  "
                  f"label={label}")
        except FileNotFoundError as e:
            print(f"[{split:5s}]  MISSING — {e}")

    print("\nCategory → label mapping (IPT paper categories starred):")
    for cat, label in sorted(CATEGORY_TO_LABEL.items(), key=lambda x: x[1]):
        star = " *" if cat in ("Airplane", "Chair", "Car") else ""
        print(f"  {label:2d}  {cat}{star}")
# """
# get_shapenet.py

# Dataset loader for ShapeNetCore.v2.PC15k — the exact dataset used by the
# IPT paper (Röell & Rieck, NeurIPS 2025).  Drop-in replacement for the
# previous torch_geometric-based loader, with an identical (pc, label) API.

# Expected directory layout after downloading and unzipping
# ShapeNetCore.v2.PC15k.zip:

#     root/
#     └── ShapeNetCore.v2.PC15k/
#         ├── 02691156/          ← synset ID for Airplane
#         │   ├── train/
#         │   │   ├── 1a04e3eab45ca15dd86060f189eb133.npy
#         │   │   └── ...
#         │   ├── val/
#         │   └── test/
#         ├── 03001627/          ← Chair
#         ├── 02958343/          ← Car
#         └── ...

# Each .npy file is a float32 array of shape (15000, 3) — a dense point
# cloud from which we subsample num_points points at random every epoch,
# giving stochastic data augmentation for free.

# Download
# --------
# https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j
# (linked from https://github.com/aidos-lab/inner-product-transforms)

# Usage
# -----
#     from get_shapenet import PointCloudShapeNetPC15k

#     train_ds = PointCloudShapeNetPC15k(
#         root       = "./data/ShapeNetCore.v2.PC15k",
#         categories = ["Airplane", "Chair", "Car"],
#         split      = "train",
#         num_points = 2048,
#     )
#     pc, label = train_ds[0]   # pc: (2048, 3),  label: int
# """

# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# SYNSET_TO_CATEGORY = {
#     "02691156": "Airplane",       
#     "02958343": "Car",            
#     "03001627": "Chair",         
# }

# CATEGORY_TO_SYNSET = {v: k for k, v in SYNSET_TO_CATEGORY.items()}

# CATEGORY_TO_LABEL = {cat: i for i, cat in enumerate(sorted(SYNSET_TO_CATEGORY.values()))}


# class PointCloudShapeNet(Dataset):
#     """
#     ShapeNetCore.v2.PC15k point cloud dataset.

#     Reads pre-computed dense point clouds (.npy, shape 15000×3) and
#     subsamples num_points points per shape on every __getitem__ call,
#     giving free stochastic augmentation during training.

#     Returns (pc, label) where:
#       pc    : Tensor (num_points, 3)  — normalised to unit sphere
#       label : int                     — integer category index

#     Parameters
#     ----------
#     root : str
#         Path to the unzipped ShapeNetCore.v2.PC15k directory,
#         i.e. the folder that contains synset-ID sub-folders like 02691156/.
#     categories : list[str] or None
#         Category names to include, e.g. ["Airplane", "Chair", "Car"].
#         None = all available categories under root.
#     split : str
#         "train", "val", or "test".  Must match sub-folder names in the
#         dataset (the PC15k release uses these exact names).
#     num_points : int
#         Number of points to subsample per shape.  The IPT paper uses 2048.
#     normalise : bool
#         Centre each point cloud and scale to unit sphere.
#         Must be True to match the IPT paper's preprocessing.
#     """

#     def __init__(
#         self,
#         root:        str        = "./data/ShapeNetCore.v2.PC15k",
#         categories:  list | None = None,
#         split:       str        = "train",
#         num_points:  int        = 2048,
#         normalise:   bool       = True,
#     ):
#         super().__init__()
#         assert split in ("train", "val", "test"), \
#             f"split must be 'train', 'val', or 'test', got '{split}'"

#         self.root       = root
#         self.split      = split
#         self.num_points = num_points
#         self.normalise  = normalise

#         if categories is not None:
#             synsets = []
#             for cat in categories:
#                 if cat not in CATEGORY_TO_SYNSET:
#                     raise ValueError(
#                         f"Unknown category '{cat}'. "
#                         f"Valid names: {sorted(CATEGORY_TO_SYNSET.keys())}"
#                     )
#                 synsets.append(CATEGORY_TO_SYNSET[cat])
#         else:
#             # Use every synset folder that exists under root
#             synsets = [
#                 d for d in os.listdir(root)
#                 if os.path.isdir(os.path.join(root, d))
#                 and d in SYNSET_TO_CATEGORY
#             ]

#         if len(synsets) == 0:
#             raise FileNotFoundError(
#                 f"No valid synset folders found under '{root}'. "
#                 "Check that the path points to the unzipped "
#                 "ShapeNetCore.v2.PC15k directory."
#             )

#         self._samples: list[tuple[str, int]] = []

#         for synset in sorted(synsets):
#             cat_name  = SYNSET_TO_CATEGORY[synset]
#             label     = CATEGORY_TO_LABEL[cat_name]
#             split_dir = os.path.join(root, synset, split)

#             if not os.path.isdir(split_dir):
#                 raise FileNotFoundError(
#                     f"Split directory not found: {split_dir}\n"
#                     f"Expected layout: root/{{synset_id}}/{{train|val|test}}/*.npy"
#                 )

#             npy_files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))

#             if len(npy_files) == 0:
#                 raise FileNotFoundError(
#                     f"No .npy files found in {split_dir}. "
#                     "Check that the zip was extracted correctly."
#                 )

#             for path in npy_files:
#                 self._samples.append((path, label))

#         if len(self._samples) == 0:
#             raise RuntimeError(
#                 f"Dataset is empty for split='{split}' and "
#                 f"categories={categories}."
#             )

#         self._categories_present = sorted({
#             SYNSET_TO_CATEGORY[s] for s in synsets
#             if SYNSET_TO_CATEGORY[s] in CATEGORY_TO_LABEL
#         })

#     def __len__(self) -> int:
#         return len(self._samples)

#     def __repr__(self) -> str:
#         return (
#             f"PointCloudShapeNet("
#             f"split={self.split}, "
#             f"n={len(self)}, "
#             f"categories={self._categories_present}, "
#             f"num_points={self.num_points})"
#         )

#     def __getitem__(self, idx: int):
#         path, label = self._samples[idx]

#         pc_full = np.load(path).astype(np.float32)   

#         N = pc_full.shape[0]
#         if N >= self.num_points:
#             idx_pts = np.random.choice(N, self.num_points, replace=False)
#         else:
#             idx_pts = np.random.choice(N, self.num_points, replace=True)

#         pc = torch.from_numpy(pc_full[idx_pts])       

#         if self.normalise:
#             pc = pc - pc.mean(dim=0, keepdim=True)
#             scale = pc.norm(dim=-1).max().clamp(min=1e-8)
#             pc = pc / scale

#         return pc, label

# if __name__ == "__main__":
#     import sys

#     root = sys.argv[1] if len(sys.argv) > 1 else "./data/ShapeNetCore.v2.PC15k"

#     print(f"Root: {root}\n")

#     for split in ("train", "val", "test"):
#         try:
#             ds = PointCloudShapeNet(
#                 root       = root,
#                 categories = ["Airplane", "Chair", "Car"],
#                 split      = split,
#                 num_points = 2048,
#             )
#             pc, label = ds[0]
#             print(f"[{split:5s}]  {len(ds):5d} shapes  |  "
#                   f"pc={tuple(pc.shape)}  "
#                   f"range=[{pc.min():.3f}, {pc.max():.3f}]  "
#                   f"label={label}")
#         except FileNotFoundError as e:
#             print(f"[{split:5s}]  MISSING — {e}")

#     print("\nCategory → label mapping (IPT paper categories starred):")
#     for cat, label in sorted(CATEGORY_TO_LABEL.items(), key=lambda x: x[1]):
#         star = " *" if cat in ("Airplane", "Chair", "Car") else ""
#         print(f"  {label:2d}  {cat}{star}")
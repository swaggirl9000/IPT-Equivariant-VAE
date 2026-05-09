import glob, numpy as np, os

root = "/home/aromanowski/IPT-Equivariant-VAE/data/ShapeNetCore.v2.PC15k"

for npy in sorted(glob.glob(f"{root}/**/train/*.npy", recursive=True))[:3]:
    try:
        pc = np.load(npy)
        print(f"OK  {os.path.relpath(npy, root)}")
        print(f"    shape={pc.shape}  dtype={pc.dtype}")
        print(f"    x∈[{pc[:,0].min():.3f}, {pc[:,0].max():.3f}]  "
              f"y∈[{pc[:,1].min():.3f}, {pc[:,1].max():.3f}]  "
              f"z∈[{pc[:,2].min():.3f}, {pc[:,2].max():.3f}]")
    except Exception as e:
        print(f"BAD {os.path.relpath(npy, root)}  →  {e}")
    print()
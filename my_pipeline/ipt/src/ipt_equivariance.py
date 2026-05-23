import argparse
import numpy as np
import torch
from lightning.fabric import Fabric

from layers.ect import EctConfig, compute_ect_point_cloud
from loaders import load_config, load_model
from spherical_harmonics import SphericalHarmonicProjection
from get_directions import get_directions


def compute_ect_chunked(x, v, radius, resolution, scale, chunk_size=64):
    """
    Chunked ECT to avoid OOM on large point clouds.
    x:   (B, N, 3)
    v:   (3, num_dirs)
    returns: (B, num_dirs, resolution)
    """
    num_dirs = v.shape[1]
    lin      = torch.linspace(-radius, radius, resolution, device=x.device)
    chunks   = []
    for start in range(0, num_dirs, chunk_size):
        v_chunk   = v[:, start:start + chunk_size]              # (3, chunk)
        nh        = x @ v_chunk                                 # (B, N, chunk)
        ecc       = torch.sigmoid(
            scale * (lin.view(1, 1, resolution) - nh.unsqueeze(-1))
        )                                                        # (B, N, chunk, R)
        ect_chunk = ecc.sum(dim=1)                              # (B, chunk, R)
        chunks.append(ect_chunk)
    ect = torch.cat(chunks, dim=1)                              # (B, num_dirs, R)
    ect = 2 * (ect / ect.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)) - 1
    return ect


def decoder_equivariance_error(
    model,
    sh_features: torch.Tensor,
    irreps,
    N: int = 200,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()
    sh_features = sh_features.to(device)        # (B, R, sh_dim)

    with torch.no_grad():
        baseline_out, _, _ = model(sh_features) # (B, R, sh_dim)

    errors = []
    for _ in range(N):
        # Sample SO(3) rotation
        rot = torch.linalg.qr(torch.randn(3, 3, device=device))[0]
        if rot.det() < 0:
            rot[:, 0] *= -1

        # Representation matrix in SH space
        D = irreps.D_from_matrix(rot.cpu()).to(device)  # (sh_dim, sh_dim)

        # Rotate input SH
        sh_rot = sh_features @ D.T          # (B, R, sh_dim)

        with torch.no_grad():
            out_rot, _, _ = model(sh_rot)   # (B, R, sh_dim)

        # Expected output under equivariance: D * baseline_out
        out_expected = baseline_out @ D.T   # (B, R, sh_dim)

        diff = (out_rot - out_expected).norm().item()
        norm = out_expected.norm().item()
        errors.append(diff / (norm + 1e-8))

    return np.array(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/equiv_vae_protein.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--N",          type=int, default=500)
    parser.add_argument("--label",      type=str, default="model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        dataconfig, _transformconfig, modelconfig,
        trainerconfig, loggerconfig,
    ) = load_config(args.config)

    if isinstance(modelconfig.ectconfig, dict):
        modelconfig.ectconfig = EctConfig(**modelconfig.ectconfig)

    ect_cfg = modelconfig.ectconfig

    fabric = Fabric(accelerator="gpu", precision=None)

    dirs, weights = get_directions(num_points=ect_cfg.num_thetas)
    dirs    = dirs.to(device)
    weights = weights.to(device)
    v       = dirs.T   # (3, num_dirs)

    sh_projection = SphericalHarmonicProjection(
        dirs=dirs, weights=weights, l_max=modelconfig.lmax
    )
    sh_projection = fabric.setup_module(sh_projection)

    # Sanity check
    assert dirs.shape[0] == sh_projection.weighted_basis.shape[0]

    # Build a small probe point cloud
    pc_probe = torch.randn(1, 512, 3, device=device)
    pc_probe = pc_probe / pc_probe.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9

    model = load_model(modelconfig)
    state = {"model": model}
    fabric.load(args.checkpoint, state)
    model = fabric.setup_module(model)
    model.eval()

    # pc → ECT → SH
    with torch.no_grad():
        ect_probe = compute_ect_chunked(
            x          = pc_probe,
            v          = v,
            radius     = ect_cfg.r,
            resolution = ect_cfg.resolution,
            scale      = ect_cfg.scale,
            chunk_size = 64,
        )   # (1, num_dirs, resolution)

        sh_probe = sh_projection(ect_probe)   # (1, resolution, sh_dim)

    print("\n--- Decoder-only equivariance test ---")
    dec_errors = decoder_equivariance_error(
        model,
        sh_probe,
        sh_projection.irreps,
        N=args.N,
        device=str(device),
    )

    print(f"\nDecoder SO(3) equivariance error ({args.label}) — {len(dec_errors)} rotations:")
    print(f"  Mean : {dec_errors.mean():.4f}")
    print(f"  Max  : {dec_errors.max():.4f}")
    print(f"  Std  : {dec_errors.std():.4f}")
    print("\n(Lower is better. A well-implemented decoder should be close to 0.)")



#DECODER
# import argparse
# import numpy as np
# import torch
# from lightning.fabric import Fabric

# from layers.ect import EctConfig, compute_ect_point_cloud
# from loaders import load_config, load_model
# from spherical_harmonics import SphericalHarmonicProjection
# from get_directions import get_directions


# def compute_ect_chunked(x, v, radius, resolution, scale, chunk_size=64):
#     """
#     Chunked ECT to avoid OOM on large point clouds.
#     x:   (B, N, 3)
#     v:   (3, num_dirs)
#     returns: (B, num_dirs, resolution)
#     """
#     num_dirs = v.shape[1]
#     lin      = torch.linspace(-radius, radius, resolution, device=x.device)
#     chunks   = []
#     for start in range(0, num_dirs, chunk_size):
#         v_chunk   = v[:, start:start + chunk_size]              # (3, chunk)
#         nh        = x @ v_chunk                                 # (B, N, chunk)
#         ecc       = torch.sigmoid(
#             scale * (lin.view(1, 1, resolution) - nh.unsqueeze(-1))
#         )                                                        # (B, N, chunk, R)
#         ect_chunk = ecc.sum(dim=1)                              # (B, chunk, R)
#         chunks.append(ect_chunk)
#     ect = torch.cat(chunks, dim=1)                              # (B, num_dirs, R)
#     ect = 2 * (ect / ect.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)) - 1
#     return ect


# def decoder_equivariance_error(
#     model,
#     sh_features: torch.Tensor,
#     irreps,
#     N: int = 200,
#     device: str = "cuda",
# ) -> np.ndarray:
#     model.eval()
#     sh_features = sh_features.to(device)

#     with torch.no_grad():
#         baseline_out, _, _ = model(sh_features)

#     errors = []
#     for _ in range(N):
#         rot = torch.linalg.qr(torch.randn(3, 3, device=device))[0]
#         if rot.det() < 0:
#             rot[:, 0] *= -1

#         D = irreps.D_from_matrix(rot.cpu()).to(device)

#         sh_rot = sh_features @ D.T

#         with torch.no_grad():
#             out_rot, _, _ = model(sh_rot)

#         out_expected = baseline_out @ rot.T

#         diff = (out_rot - out_expected).norm().item()
#         norm = out_expected.norm().item()
#         errors.append(diff / (norm + 1e-8))

#     return np.array(errors)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config",     default="configs/equiv_vae_protein.yaml")
#     parser.add_argument("--checkpoint", required=True)
#     parser.add_argument("--N",          type=int, default=500)
#     parser.add_argument("--label",      type=str, default="model")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     (
#         dataconfig, _transformconfig, modelconfig,
#         trainerconfig, loggerconfig,
#     ) = load_config(args.config)

#     if isinstance(modelconfig.ectconfig, dict):
#         modelconfig.ectconfig = EctConfig(**modelconfig.ectconfig)

#     ect_cfg = modelconfig.ectconfig

#     fabric = Fabric(accelerator="gpu", precision=None)

#     dirs, weights = get_directions(num_points=ect_cfg.num_thetas)
#     dirs    = dirs.to(device)
#     weights = weights.to(device)
#     v       = dirs.T   # (3, num_dirs)

#     sh_projection = SphericalHarmonicProjection(
#         dirs=dirs, weights=weights, l_max=modelconfig.l_max
#     )
#     sh_projection = fabric.setup_module(sh_projection)

#     # Sanity check
#     assert dirs.shape[0] == sh_projection.weighted_basis.shape[0]

#     # Build a small probe point cloud
#     pc_probe = torch.randn(1, 512, 3, device=device)
#     pc_probe = pc_probe / pc_probe.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9

#     model = load_model(modelconfig)
#     state = {"model": model}
#     fabric.load(args.checkpoint, state)
#     model = fabric.setup_module(model)
#     model.eval()

#     # pc → ECT → SH
#     with torch.no_grad():
#         ect_probe = compute_ect_chunked(
#             x          = pc_probe,
#             v          = v,
#             radius     = ect_cfg.r,
#             resolution = ect_cfg.resolution,
#             scale      = ect_cfg.scale,
#             chunk_size = 64,
#         )   # (1, num_dirs, resolution)

#         sh_probe = sh_projection(ect_probe)   # (1, resolution, sh_dim)

#     print("\n--- Decoder-only equivariance test ---")
#     dec_errors = decoder_equivariance_error(
#         model,
#         sh_probe,
#         sh_projection.irreps,
#         N=args.N,
#         device=str(device),
#     )

#     print(f"\nDecoder SO(3) equivariance error ({args.label}) — {len(dec_errors)} rotations:")
#     print(f"  Mean : {dec_errors.mean():.4f}")
#     print(f"  Max  : {dec_errors.max():.4f}")
#     print(f"  Std  : {dec_errors.std():.4f}")
#     print("\n(Lower is better. A well-implemented decoder should be close to 0.)")

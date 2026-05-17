import torch
import numpy as np
from e3nn import o3

from get_directions import get_directions
from pipeline import IPTVAEPipeline
from train import load_checkpoint
# from get_shapenet import PointCloudShapeNet
from get_protiens import ProteinNeighborhoods  


def equivariance_error(
    model:  IPTVAEPipeline,
    pc:     torch.Tensor,
    N:      int = 100,
    device: str = "cpu",
) -> list:
    """
    Test equivariance of the full pipeline.

    For each of N random SO(3) rotations g:
        F(g · pc)   rotate input, then run pipeline
        g · F(pc)   run pipeline, then rotate output

    Equivariance error = ||F(g·pc) - g·F(pc)|| / ||F(pc)||
    """
    model.eval()
    pc = pc.to(device)

    with torch.no_grad():
        c_pred, _, _, _, _, _ = model(pc)

    l_max  = model.l_max
    errors = []

    for _ in range(N):
        rot_cpu = o3.rand_matrix()       
        rot_gpu = rot_cpu.to(device)       

        # F(g · pc) 
        pc_rotated = pc @ rot_gpu.T
        with torch.no_grad():
            c_pred_rot, _, _, _, _, _ = model(pc_rotated)

        # g · F(pc) 
        c_rotated_output = torch.zeros_like(c_pred)
        sh_idx = 0
        for l in range(l_max + 1):
            m         = 2 * l + 1
            irrep_str = f"1x{l}{'e' if l % 2 == 0 else 'o'}"
            D_l       = o3.Irreps(irrep_str).D_from_matrix(rot_cpu)
            D_l       = D_l.to(device)                                

            block         = c_pred[:, sh_idx:sh_idx + m, :]
            rotated_block = torch.einsum("ij, bjr -> bir", D_l, block)
            c_rotated_output[:, sh_idx:sh_idx + m, :] = rotated_block
            sh_idx += m

        norm_output = c_pred.norm().item()
        norm_diff   = (c_pred_rot - c_rotated_output).norm().item()
        errors.append(norm_diff / (norm_output + 1e-8))

    return errors


def run_equivariance_test(
    model:        IPTVAEPipeline,
    device:       str = "cpu",
    N:            int = 100,
    n_trials:     int = 5,
    from_dataset: torch.utils.data.Dataset = None,
) -> np.ndarray:

    model.eval()
    all_errors = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")

        if from_dataset is not None:
            idx = torch.randint(len(from_dataset), (1,)).item()
            pc, _ = from_dataset[idx]
            pc = pc.unsqueeze(0)
        else:
            pc = torch.randn(1, 256, 3)
            pc = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9

        errors = equivariance_error(model, pc, N=N, device=device)
        all_errors.extend(errors)

    all_errors = np.array(all_errors)
    print(f"\nSO(3) equivariance test — {len(all_errors)} random rotations:")
    print(f"  Mean error : {all_errors.mean():.6f}")
    print(f"  Max  error : {all_errors.max():.6f}")
    print(f"  Std  error : {all_errors.std():.6f}")

    return all_errors


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = [
        "/gpfs/home3/aromanowski/IPT-Equivariant-VAE/checkpoint_ipt_vae_proteins_lmax4_R8_leb59.pt",
    ]

    dataset = ProteinNeighborhoods(
        processed_dir = "/gpfs/home3/aromanowski/IPT-Equivariant-VAE/data/protein/processed",
        split         = "test",
        num_points    = 512,   
    )
    results = {}

    for ckpt_path in checkpoints:
        print("=" * 50)
        print(f"Checkpoint: {ckpt_path}")

        state_dict, ckpt_config = load_checkpoint(ckpt_path, device)

        l_max         = ckpt_config.get("l_max", 2)
        R             = ckpt_config.get("R", 8)
        lebedev_order = ckpt_config.get("lebedev_order", 19)

        print(f"  l_max={l_max} | R={R} | lebedev={lebedev_order}")
        print("=" * 50)

        dirs, weights = get_directions(lebedev_order)
        dirs, weights = dirs.to(device), weights.to(device)

        model = IPTVAEPipeline(dirs, weights, l_max=l_max, R=R).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        errors = run_equivariance_test(
            model,
            device       = str(device),
            N            = 100,
            n_trials     = 5,
            from_dataset = dataset,
        )

        results[ckpt_path] = {
            "mean": float(errors.mean()),
            "max":  float(errors.max()),
            "std":  float(errors.std()),
        }

    print("\nSUMMARY")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name}")
        print(f"  mean={r['mean']:.6f} | max={r['max']:.6f} | std={r['std']:.6f}")
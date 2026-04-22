import torch
import numpy as np
import matplotlib.pyplot as plt
from e3nn import o3

from get_directions import get_directions
from pipeline import IPTVAEPipeline

def equivariance_error(
    model:      IPTVAEPipeline,
    pc:         torch.Tensor,   
    N:          int = 100,
    device:     str = "cpu",
) -> tuple[list, list]:
    """
    Test equivariance of the full pipeline.

    For each rotation g:
        F(g · pc)      rotate input, then run pipeline
        g · F(pc)      run pipeline, then rotate output 

    Equivariance error = ||F(g·pc) - g·F(pc)|| / ||F(pc)||

    Returns
    -------
    angles : list of rotation angles in radians
    errors : list of relative equivariance errors
    """
    model.eval()
    pc = pc.to(device)
    
    with torch.no_grad():
        c_pred, _, _, _, _, _ = model(pc)
    
    l_max = model.l_max
    R = model.R
    F = (l_max + 1) ** 2
    
    angles = []
    errors = []
    
    #random rotation
    for i in range(N+1):
        theta = i / N * 2 * np.pi
        rot = o3.matrix_z(torch.tensor(theta))
        
        #F(g·pc)
        pc_rotated = (pc @ rot.T.to(device))
        with torch.no_grad():
            c_pred_rot, _, _, _, _, _ = model(pc_rotated)
            
        #g·F(pc)
        c_rotated_output = torch.zeros_like(c_pred)
        sh_idx = 0
        for l in range(l_max + 1):
            m = 2 * l + 1
            irrep_str = f"1x{l}{'e' if l % 2 == 0 else 'o'}"
            D_l = o3.Irreps(irrep_str).D_from_matrix(rot)   
            D_l = D_l.to(device)

            block = c_pred[:, sh_idx:sh_idx+m, :]         
            rotated_block = torch.einsum("ij, bjr -> bir", D_l, block)
            c_rotated_output[:, sh_idx:sh_idx+m, :] = rotated_block
            sh_idx += m
        
        norm_output   = c_pred.norm().item()
        norm_diff     = (c_pred_rot - c_rotated_output).norm().item()
        rel_error     = norm_diff / (norm_output + 1e-8)

        angles.append(theta)
        errors.append(rel_error)
        
    return angles, errors

def run_equivariance_test(
    model:    IPTVAEPipeline,
    device:   str = "cpu",
    N:        int = 100,
    n_trials: int = 5,
):
    model.eval()
    errors = []
    
    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...")
        pc = torch.randn(1, 256, 3)
        pc[:, :, 2] = 0.0                 
        pc = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9
        
        angles, error = equivariance_error(model, pc, N=N, device=device)
        errors.append(error)
    
    errors = np.array(errors) 
    mean_err   = errors.mean(axis=0)
    std_err    = errors.std(axis=0)
    
    print(f"\nMean equivariance error across all angles: {mean_err.mean():.6f}")
    print(f"Max  equivariance error across all angles: {mean_err.max():.6f}")

    return angles, errors

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirs, weights = get_directions(110)
    dirs, weights = dirs.to(device), weights.to(device)

    model = IPTVAEPipeline(dirs, weights, l_max=6, R=8).to(device)

    angles, all_errors = run_equivariance_test(
        model, device=str(device), N=100, n_trials=5
    )
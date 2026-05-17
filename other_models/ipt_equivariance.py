import torch
import numpy as np
from lightning.fabric import Fabric
from torch_geometric.data import Data
from layers.ect import EctLayer, EctConfig
from loaders import load_config, load_model
import argparse
 
def equivariance_error(model, ect_layer, pc, N=100, device="cuda"):
    model.eval()
    pc = pc.to(device)
 
    def run(points):
        b, n, _ = points.shape
        idx = torch.arange(b, device=device).repeat_interleave(n)
        batch = Data(x=points.reshape(-1, 3))
        ect = ect_layer(batch, idx).unsqueeze(1)
        with torch.no_grad():
            output = model(ect)
            if isinstance(output, tuple):
                return output[0]
            return output
 
    baseline = run(pc)
    errors = []
 
    for _ in range(N):
        rot = torch.linalg.qr(torch.randn(3, 3))[0].to(device)
        pc_rot = pc @ rot.T
        recon_rot = run(pc_rot)

        diff = (recon_rot - baseline).norm().item()
        norm = baseline.norm().item()
        errors.append(diff / (norm + 1e-8))
 
    return np.array(errors)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vae_protein.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--label", type=str, default="model")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    dataconfig, transformconfig, modelconfig, trainerconfig, loggerconfig = load_config(args.config)
 
    print("ectconfig.num_thetas:", modelconfig.ectconfig.num_thetas)
    print("ectlossconfig.num_thetas:", modelconfig.ectlossconfig.num_thetas)
    fabric = Fabric(accelerator="gpu", precision=None)
    model = load_model(modelconfig)
 
    state = {"model": model}
    fabric.load(args.checkpoint, state)
    model = fabric.setup_module(model)
    model.eval()
 
    v = torch.randn(3, modelconfig.ectconfig.num_thetas)
    v /= torch.norm(v, dim=0)
    v = v.to(device)
    ect_layer = EctLayer(config=modelconfig.ectconfig, v=v)
    ect_layer = fabric.setup_module(ect_layer)
 
    all_errors = []
    for trial in range(args.n_trials):
        print(f"Trial {trial+1}/{args.n_trials}...")
        pc = torch.randn(1, 512, 3)
        pc = pc / pc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.9
        errors = equivariance_error(model, ect_layer, pc, N=args.N, device=str(device))
        all_errors.extend(errors)
 
    all_errors = np.array(all_errors)
    print(f"\nSO(3) equivariance error ({args.label}) — {len(all_errors)} rotations:")
    print(f"  Mean : {all_errors.mean():.4f}")
    print(f"  Max  : {all_errors.max():.4f}")
    print(f"  Std  : {all_errors.std():.4f}")
    print("\n(Higher = less equivariant. Use this as your baseline number to beat.)")
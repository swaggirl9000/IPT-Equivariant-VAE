import argparse
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf    
from trainers.vae_trainer import VAETrainer
from shapenet_datamodule import ShapeNetDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n",    type=int, default=8)
    parser.add_argument("--out",  default="/home/aromanowski/IPT-Equivariant-VAE/verify_ect.png")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAETrainer.load_from_checkpoint(args.ckpt, cfg=cfg).to(device).eval()

    data_cfg = OmegaConf.create({          
        "data_dir":   "/home/aromanowski/IPT-Equivariant-VAE/data/ShapeNetCore.v2.PC15k/02691156",
        "batch_size": 8,
        "num_workers": 0,
        "n_points":   2048,
    })
    dm = ShapeNetDataModule(data_cfg)
    dm.setup("test")

    batch = next(iter(dm.test_dataloader()))
    pc  = batch[0] if isinstance(batch, (list, tuple)) else batch
    pc  = pc.to(model.device)                              
    ect = model.ecttransform(pc).unsqueeze(1)

    with torch.no_grad():
        recon, mu, log_var = model.model(ect)
    print("ECT   min/max/mean:", ect.min().item(), ect.max().item(), ect.mean().item())
    print("Recon min/max/mean:", recon.min().item(), recon.max().item(), recon.mean().item())
    print("Are they the same?", torch.allclose(recon[0], recon[1], atol=1e-3))  # True = collapsed
    n = min(args.n, ect.size(0))
    fig, axes = plt.subplots(nrows=3, ncols=n, figsize=(3 * n, 9))

    for i in range(n):
        gt_img    = ect[i].squeeze().cpu().numpy()
        recon_img = recon[i].squeeze().cpu().numpy()
        error_img = abs(gt_img - recon_img)

        axes[0, i].imshow(gt_img,    cmap="bone", vmin=-0.5, vmax=1.5)
        axes[1, i].imshow(recon_img, cmap="bone", vmin=-0.5, vmax=1.5)
        axes[2, i].imshow(error_img, cmap="hot",  vmin=0,    vmax=1)
        for row in range(3):
            axes[row, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    axes[2, 0].set_ylabel("Error |gt-recon|")

    plt.suptitle(f"ECT verification — {args.ckpt}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
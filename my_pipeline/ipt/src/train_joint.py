"""
Joint end-to-end training of the equivariant VAE + equivariant encoder.
Usage:
    python train_joint.py \
        --vae_config     configs/equiv_vae.yaml \
        --encoder_config configs/encoder_airplane.yaml \
"""

import argparse
import os

import torch
import torch.nn.functional as F
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from tqdm import tqdm
from layers.ect import EctConfig

from loaders import (
    load_config,
    load_datamodule,
    load_logger,
    load_model,
)
from metrics.loss import chamfer
from spherical_harmonics import SphericalHarmonicProjection
from transforms.ecttransform import Transform, TransformConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_ect_chunked(
    x: torch.Tensor,
    v: torch.Tensor,
    radius: float,
    resolution: int,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    lin    = torch.linspace(-radius, radius, resolution, device=x.device)
    chunks = []
    for start in range(0, v.shape[1], chunk_size):
        v_c   = v[:, start:start + chunk_size]                
        nh    = x @ v_c                                     
        ecc   = torch.sigmoid(
            scale * (lin.view(1, 1, 1, resolution) - nh.unsqueeze(-1))
        )                                                  
        chunks.append(ecc.sum(dim=1))                         
    ect = torch.cat(chunks, dim=1)                           
    ect = 2 * (ect / ect.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)) - 1
    return ect


def beta_schedule(epoch: int, period: int, beta_min: float, beta_max: float) -> float:
    if period <= 0:
        return beta_max
    t = min(epoch / period, 1.0)
    return beta_min + 0.5 * (beta_max - beta_min) * (1 - torch.cos(torch.tensor(t * 3.14159)).item())

def weighted_sh_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    l_max: int,
) -> torch.Tensor:
    loss   = torch.tensor(0.0, device=pred.device)
    offset = 0
    for l in range(l_max + 1):
        m   = 2 * l + 1
        w   = 1.0 / m   #
        loss = loss + w * F.mse_loss(
            pred[...,   offset:offset + m],
            target[..., offset:offset + m],
        )
        offset += m
    return loss

def plot_recon_3d(pcs_recon, pcs_gt, num_pc=8, filename="plot.png"):
    recon  = pcs_recon.detach().cpu().numpy()
    gt     = pcs_gt.detach().cpu().numpy()
    n_plot = min(num_pc, len(recon), len(gt))
    fig    = plt.figure(figsize=(4 * n_plot, 8))
    for i in range(n_plot):
        ax = fig.add_subplot(2, n_plot, i + 1, projection="3d")
        ax.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2], c="#1f77b4", s=3, alpha=0.6)
        ax.set_title(f"GT {i}")
        ax.axis("off")
        ax = fig.add_subplot(2, n_plot, n_plot + i + 1, projection="3d")
        ax.scatter(recon[i, :, 0], recon[i, :, 1], recon[i, :, 2], c="#d62728", s=3, alpha=0.6)
        ax.set_title(f"Recon {i}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {filename}")

def train(
    fabric,
    dataloader,
    vae_model,
    enc_model,
    optimizer_vae,
    optimizer_enc,
    sh_transform,
    losstransform,
    dirs,
    vae_modelconfig,
    trainerconfig,
    loggerconfig,
    no_progressbar,
    results_base_dir,
):
    ect_cfg    = vae_modelconfig.ectconfig
    v          = dirs.T.to(fabric.device)        
    chunk_size = getattr(ect_cfg, "chunk_size", 64)

    step = 0
    for epoch in range(trainerconfig.max_epochs):
        vae_model.eval()
        enc_model.train()

        for pc in tqdm(dataloader, disable=no_progressbar):
            optimizer_enc.zero_grad(set_to_none=True)

            # PC → ECT → SH  
            with torch.no_grad():
                ect   = compute_ect_chunked(
                    x=pc, v=v,
                    radius=ect_cfg.r, resolution=ect_cfg.resolution,
                    scale=ect_cfg.scale, chunk_size=chunk_size,
                )
                sh_gt = sh_transform(ect)

                sh_recon, _, _ = vae_model(sh_gt)

            pc_recon  = enc_model(sh_recon)
            cd_loss   = chamfer(pc_recon, pc)
            ipt_gt    = losstransform(pc)
            ipt_recon = losstransform(pc_recon)
            ipt_loss  = F.mse_loss(ipt_gt, ipt_recon)

            loss = 1.0 * cd_loss + 10.0 * ipt_loss

            if torch.isnan(loss):
                print(f"[warn] NaN loss at step {step} — skipping step")
                optimizer_enc.zero_grad(set_to_none=True)
                step += 1
                continue

            fabric.backward(loss)

            torch.nn.utils.clip_grad_norm_(enc_model.parameters(), max_norm=1.0)

            optimizer_enc.step()

            if step % 10 == 0:
                print(
                    f"epoch {epoch:04d}  step {step:06d} | "
                    f"loss {loss.item():.4f}  "
                    f"cd {cd_loss.item():.4f}  "
                    f"ipt {ipt_loss.item():.4f}"
                )
            step += 1

        if epoch % trainerconfig.checkpoint_interval == 0:
            fabric.save(f"{results_base_dir}/enc_model.ckpt", {"model": enc_model})
            plot_recon_3d(
                pc_recon[:8].detach(), pc[:8].detach(),
                filename=f"{results_base_dir}/recon_{epoch:04d}.png",
            )

    fabric.save(f"{results_base_dir}/enc_model.ckpt", {"model": enc_model})

@torch.no_grad()
def evaluate(
    fabric,
    valdataloader,
    vae_model,
    enc_model,
    sh_transform,
    dirs,
    vae_modelconfig,
    loggerconfig,
    results_base_dir,
):
    ect_cfg    = vae_modelconfig.ectconfig
    v          = dirs.T.to(fabric.device)
    chunk_size = getattr(ect_cfg, "chunk_size", 64)

    vae_model.eval()
    enc_model.eval()

    recon_pcs_all, gt_pcs_all = [], []

    for pc in tqdm(valdataloader):
        ect   = compute_ect_chunked(
            x=pc, v=v,
            radius=ect_cfg.r, resolution=ect_cfg.resolution,
            scale=ect_cfg.scale, chunk_size=chunk_size,
        )
        sh_gt             = sh_transform(ect)
        
        pc_recon          = enc_model(sh_gt)

        recon_pcs_all.append(pc_recon.cpu())
        gt_pcs_all.append(pc.cpu())

    recon_pcs = torch.cat(recon_pcs_all)
    gt_pcs    = torch.cat(gt_pcs_all)

    torch.save(recon_pcs, f"{results_base_dir}/recon_pcs.pt")
    torch.save(gt_pcs,    f"{results_base_dir}/gt_pcs.pt")

    plot_recon_3d(
        recon_pcs[:8], gt_pcs[:8],
        filename=f"{results_base_dir}/pcs_final.png",
    )
    print(f"Saved evaluation outputs to {results_base_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Joint VAE + Encoder training")
    parser.add_argument("--vae_config",     required=True,  type=str)
    parser.add_argument("--encoder_config", required=True,  type=str)
    parser.add_argument("--vae_checkpoint", required=True,  type=str,
                        help="Path to pre-trained VAE checkpoint, e.g. "
                             "results/equivariant_vae_protein/model.ckpt")
    parser.add_argument("--resume_enc",     default=False,  action="store_true")
    parser.add_argument("--compile",        default=False,  action="store_true")
    parser.add_argument("--dev",            default=False,  action="store_true")
    parser.add_argument("--no-progressbar", default=False,  action="store_true")
    args = parser.parse_args()

    (dataconfig, _, enc_modelconfig, trainerconfig, _) = load_config(args.encoder_config)
    (_, _, vae_modelconfig, _, loggerconfig)           = load_config(args.vae_config)

    for cfg in (enc_modelconfig, vae_modelconfig):
        for field in ("ectconfig", "ectlossconfig"):
            val = getattr(cfg, field, None)
            if isinstance(val, dict):
                setattr(cfg, field, EctConfig(**val))

    results_base_dir = "results_joint"
    if args.dev:
        trainerconfig.max_epochs = 10
        results_base_dir += "_dev"
    results_base_dir += f"/{loggerconfig.results_dir}"
    os.makedirs(results_base_dir, exist_ok=True)

    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )
    seed_everything(trainerconfig.seed)
    logger = load_logger(loggerconfig)

    dm            = load_datamodule(dataconfig, dev=args.dev)
    dataloader    = fabric.setup_dataloaders(dm.train_dataloader)
    valdataloader = fabric.setup_dataloaders(dm.val_dataloader)

    from transforms.ecttransform import Transform as EctTransform
    ect_transform_layer = EctTransform(
        TransformConfig(module="", ectconfig=enc_modelconfig.ectconfig)
    )
    v_raw = ect_transform_layer.v
    dirs  = (v_raw.T if v_raw.shape[0] == 3 else v_raw).to(fabric.device)
    assert dirs.shape[1] == 3, f"Expected (num_dirs, 3), got {dirs.shape}"

    weights      = torch.ones(dirs.shape[0], device=fabric.device)
    sh_transform = SphericalHarmonicProjection(
        dirs=dirs, weights=weights, l_max=enc_modelconfig.lmax
    )
    sh_transform = fabric.setup_module(sh_transform)

    losstransform = EctTransform(
        TransformConfig(module="", ectconfig=enc_modelconfig.ectlossconfig)
    )
    losstransform = fabric.setup_module(losstransform)

    vae_model = load_model(vae_modelconfig)
    enc_model = load_model(enc_modelconfig)

    print(f"Loading pre-trained VAE from {args.vae_checkpoint}")
    fabric.load(args.vae_checkpoint, {"model": vae_model})
    for p in vae_model.parameters():
        p.requires_grad_(False)

    if args.resume_enc:
        ckpt = f"{results_base_dir}/enc_model.ckpt"
        print(f"Resuming encoder from {ckpt}")
        fabric.load(ckpt, {"model": enc_model})

    if args.compile:
        vae_model = torch.compile(vae_model)
        enc_model = torch.compile(enc_model)

    optimizer_enc = Adam(enc_model.parameters(), lr=enc_modelconfig.learning_rate, betas=(0.5, 0.999))

    vae_model = fabric.setup_module(vae_model)
    enc_model, optimizer_enc = fabric.setup(enc_model, optimizer_enc)

    train(
        fabric        = fabric,
        dataloader    = dataloader,
        vae_model     = vae_model,
        enc_model     = enc_model,
        optimizer_vae = None,
        optimizer_enc = optimizer_enc,
        sh_transform  = sh_transform,
        losstransform = losstransform,
        dirs          = dirs,
        vae_modelconfig   = vae_modelconfig,
        trainerconfig     = trainerconfig,
        loggerconfig      = loggerconfig,
        no_progressbar    = args.no_progressbar,
        results_base_dir  = results_base_dir,
    )

    evaluate(
        fabric           = fabric,
        valdataloader    = valdataloader,
        vae_model        = vae_model,
        enc_model        = enc_model,
        sh_transform     = sh_transform,
        dirs             = dirs,
        vae_modelconfig  = vae_modelconfig,
        loggerconfig     = loggerconfig,
        results_base_dir = results_base_dir,
    )


if __name__ == "__main__":
    main()
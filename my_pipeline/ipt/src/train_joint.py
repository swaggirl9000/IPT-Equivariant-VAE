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

            pc_recon = enc_model(sh_gt)         
            cd_loss = chamfer(pc_recon, pc)
            ipt_gt    = losstransform(pc)
            ipt_recon = losstransform(pc_recon)
            ipt_loss  = F.mse_loss(ipt_gt, ipt_recon)

            loss = (
                1.0  * cd_loss
                + 10.0 * ipt_loss
            )

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
# def train(
#     fabric,
#     dataloader,
#     vae_model,
#     enc_model,
#     optimizer_vae,
#     optimizer_enc,
#     sh_transform,
#     losstransform,
#     dirs,
#     vae_modelconfig,
#     trainerconfig,
#     loggerconfig,
#     no_progressbar,
#     results_base_dir,
# ):
#     ect_cfg    = vae_modelconfig.ectconfig
#     v          = dirs.T.to(fabric.device)          # (3, num_dirs)
#     chunk_size = getattr(ect_cfg, "chunk_size", 64)

#     step = 0
#     for epoch in range(trainerconfig.max_epochs):
#         vae_model.train()
#         enc_model.train()

#         beta = beta_schedule(
#             epoch,
#             vae_modelconfig.beta_period,
#             vae_modelconfig.beta_min,
#             vae_modelconfig.beta_max,
#         )

#         for pc in tqdm(dataloader, disable=no_progressbar):
#             optimizer_vae.zero_grad(set_to_none=True)
#             optimizer_enc.zero_grad(set_to_none=True)

#             # ── Step 1: PC → ECT → SH  (fixed transforms, no grad) ─────── #
#             with torch.no_grad():
#                 ect   = compute_ect_chunked(
#                     x=pc, v=v,
#                     radius=ect_cfg.r, resolution=ect_cfg.resolution,
#                     scale=ect_cfg.scale, chunk_size=chunk_size,
#                 )                                  # (B, num_dirs, resolution)
#                 sh_gt = sh_transform(ect)          # (B, resolution, sh_dim)

#             # ── Step 2: SH → VAE → SH_recon  (VAE params get grads) ────── #
#             sh_recon, mu, logvar = vae_model(sh_gt)   # (B, resolution, sh_dim)

#             # ── Step 3: SH_recon → Encoder → PC_recon ───────────────────── #
#             # Gradient flows through enc_model AND back into vae_model here.
#             pc_recon = enc_model(sh_recon)             # (B, num_pts, 3)

#             # ── Losses ──────────────────────────────────────────────────── #

#             logvar_exp  = vae_model.expand_logvar(logvar.clamp(min=-10, max=10))
#             kld_per_dim = -0.5 * (1 + logvar_exp - mu ** 2 - logvar_exp.exp())  # (B, D)
#             # Free-bits: each latent dimension is allowed FREE_BITS nats of
#             # "free" KL before we penalise it.  This prevents posterior collapse
#             # on low-information dimensions without blocking encoding on others.
#             FREE_BITS = 0.5
#             kld_loss  = torch.clamp(kld_per_dim, min=FREE_BITS).sum(dim=1).mean()
#             # NOTE: we do NOT divide by latent_dim here so that beta is on a
#             # meaningful scale relative to cd/mse.  The reported kld_loss is
#             # the per-sample sum (in nats); beta should be O(1e-4)–O(1e-2).

#             # VAE: SH reconstruction fidelity (per-degree weighted to prevent
#             # l=0 scalars from drowning the signal from higher-l structure)
#             mse_loss = weighted_sh_mse(sh_recon, sh_gt, vae_model.l_max)

#             # Encoder: Chamfer distance on point clouds
#             cd_loss = chamfer(pc_recon, pc)

#             # Encoder: IPT consistency (keeps geometry correct)
#             ipt_gt    = losstransform(pc)
#             ipt_recon = losstransform(pc_recon)
#             ipt_loss  = F.mse_loss(ipt_gt, ipt_recon)

#             # Combined — single backward through both models
#             # kld_loss is now the per-sample sum (not per-dim), so beta should
#             # be O(1e-4)–O(1e-2).  mse_loss is l-weighted so its scale is
#             # similar to the unweighted value.
#             loss = (
#                 0.1  * mse_loss
#                 + beta * kld_loss
#                 + 1.0  * cd_loss
#                 + 10.0 * ipt_loss
#             )

#             # Skip NaN steps (can occur early in training before stabilising)
#             if torch.isnan(loss):
#                 print(
#                     f"[warn] NaN loss at step {step} — "
#                     f"mse={mse_loss.item():.4f}  kld={kld_loss.item():.4f}  "
#                     f"cd={cd_loss.item():.4f}  ipt={ipt_loss.item():.4f}  "
#                     f"— skipping step"
#                 )
#                 optimizer_vae.zero_grad(set_to_none=True)
#                 optimizer_enc.zero_grad(set_to_none=True)
#                 step += 1
#                 continue

#             fabric.backward(loss)

#             # Gradient clipping — prevents exploding gradients in joint backprop
#             torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(enc_model.parameters(), max_norm=1.0)

#             optimizer_vae.step()
#             optimizer_enc.step()

#             if step % 10 == 0:
#                 print(
#                     f"epoch {epoch:04d}  step {step:06d} | "
#                     f"loss {loss.item():.4f}  "
#                     f"cd {cd_loss.item():.4f}  "
#                     f"ipt {ipt_loss.item():.4f}  "
#                     f"mse {mse_loss.item():.4f}  "
#                     f"kld {kld_loss.item():.4f}  "
#                     f"beta {beta:.8f}"
#                 )
#             step += 1

#         # ── Checkpoint + visualisation ───────────────────────────────────── #
#         if epoch % trainerconfig.checkpoint_interval == 0:
#             fabric.save(f"{results_base_dir}/vae_model.ckpt",     {"model": vae_model})
#             fabric.save(f"{results_base_dir}/enc_model.ckpt",     {"model": enc_model})
#             plot_recon_3d(
#                 pc_recon[:8].detach(), pc[:8].detach(),
#                 filename=f"{results_base_dir}/recon_{epoch:04d}.png",
#             )

#     # Final checkpoints
#     fabric.save(f"{results_base_dir}/vae_model.ckpt", {"model": vae_model})
#     fabric.save(f"{results_base_dir}/enc_model.ckpt", {"model": enc_model})


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
# @torch.no_grad()
# def evaluate(
#     fabric,
#     valdataloader,
#     vae_model,
#     enc_model,
#     sh_transform,
#     dirs,
#     vae_modelconfig,
#     loggerconfig,
#     results_base_dir,
# ):
#     ect_cfg    = vae_modelconfig.ectconfig
#     v          = dirs.T.to(fabric.device)
#     chunk_size = getattr(ect_cfg, "chunk_size", 64)

#     vae_model.eval()
#     enc_model.eval()

#     recon_sh_all, gt_sh_all   = [], []
#     recon_pcs_all, gt_pcs_all = [], []

#     for pc in tqdm(valdataloader):
#         ect   = compute_ect_chunked(
#             x=pc, v=v,
#             radius=ect_cfg.r, resolution=ect_cfg.resolution,
#             scale=ect_cfg.scale, chunk_size=chunk_size,
#         )
#         sh_gt             = sh_transform(ect)
#         sh_recon, _, _    = vae_model(sh_gt)
#         pc_recon          = enc_model(sh_recon)

#         recon_sh_all.append(sh_recon.cpu())
#         gt_sh_all.append(sh_gt.cpu())
#         recon_pcs_all.append(pc_recon.cpu())
#         gt_pcs_all.append(pc.cpu())

#     recon_sh  = torch.cat(recon_sh_all)
#     gt_sh     = torch.cat(gt_sh_all)
#     recon_pcs = torch.cat(recon_pcs_all)
#     gt_pcs    = torch.cat(gt_pcs_all)

#     torch.save(recon_sh,  f"{results_base_dir}/recon_sh.pt")
#     torch.save(gt_sh,     f"{results_base_dir}/gt_sh.pt")
#     torch.save(recon_pcs, f"{results_base_dir}/recon_pcs.pt")
#     torch.save(gt_pcs,    f"{results_base_dir}/gt_pcs.pt")

#     plot_recon_3d(
#         recon_pcs[:8], gt_pcs[:8],
#         filename=f"{results_base_dir}/pcs_final.png",
#     )
#     print(f"Saved evaluation outputs to {results_base_dir}/")


# # ─── Entry point ──────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Joint VAE + Encoder training")
    parser.add_argument("--vae_config",     required=True,  type=str)
    parser.add_argument("--encoder_config", required=True,  type=str)
    parser.add_argument("--resume_vae",     default=False,  action="store_true")
    parser.add_argument("--resume_enc",     default=False,  action="store_true")
    parser.add_argument("--compile",        default=False,  action="store_true")
    parser.add_argument("--dev",            default=False,  action="store_true")
    parser.add_argument("--no-progressbar", default=False,  action="store_true")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────── #
    (dataconfig, _, enc_modelconfig, trainerconfig, _) = load_config(args.encoder_config)
    (_, _, vae_modelconfig, _, loggerconfig)           = load_config(args.vae_config)

    # Normalise ectconfig dicts → objects
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

    # ── Fabric / seed ─────────────────────────────────────────────────────── #
    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )
    seed_everything(trainerconfig.seed)
    logger = load_logger(loggerconfig)

    # ── Data ──────────────────────────────────────────────────────────────── #
    dm            = load_datamodule(dataconfig, dev=args.dev)
    dataloader    = fabric.setup_dataloaders(dm.train_dataloader)
    valdataloader = fabric.setup_dataloaders(dm.val_dataloader)

    # ── Direction vectors + SH projection ────────────────────────────────── #
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

    # ── IPT loss transform ────────────────────────────────────────────────── #
    losstransform = EctTransform(
        TransformConfig(module="", ectconfig=enc_modelconfig.ectlossconfig)
    )
    losstransform = fabric.setup_module(losstransform)

    # ── Models ────────────────────────────────────────────────────────────── #
    vae_model = load_model(vae_modelconfig)
    enc_model = load_model(enc_modelconfig)

    if args.resume_vae:
        ckpt = f"{results_base_dir}/vae_model.ckpt"
        print(f"Resuming VAE from {ckpt}")
        fabric.load(ckpt, {"model": vae_model})

    if args.resume_enc:
        ckpt = f"{results_base_dir}/enc_model.ckpt"
        print(f"Resuming encoder from {ckpt}")
        fabric.load(ckpt, {"model": enc_model})

    if args.compile:
        vae_model = torch.compile(vae_model)
        enc_model = torch.compile(enc_model)

    # ── Optimisers ────────────────────────────────────────────────────────── #
    # Separate optimisers so you can easily use different learning rates.
    # Gradient from the point-cloud loss flows through enc_model back into
    # vae_model in a single backward pass because sh_recon connects them.
    # Halved vs single-model training — joint backprop is less stable
    optimizer_vae = Adam(vae_model.parameters(), lr=vae_modelconfig.lr * 0.5, betas=(0.5, 0.999))
    optimizer_enc = Adam(enc_model.parameters(), lr=enc_modelconfig.learning_rate * 0.5)

    vae_model, optimizer_vae = fabric.setup(vae_model, optimizer_vae)
    enc_model, optimizer_enc = fabric.setup(enc_model, optimizer_enc)

    # ── Train ─────────────────────────────────────────────────────────────── #
    train(
        fabric        = fabric,
        dataloader    = dataloader,
        vae_model     = vae_model,
        enc_model     = enc_model,
        optimizer_vae = optimizer_vae,
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

    # ── Evaluate ──────────────────────────────────────────────────────────── #
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
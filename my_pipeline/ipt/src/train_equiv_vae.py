import argparse
import os

import torch
import torchvision
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from torchvision.utils import make_grid
from tqdm import tqdm

from spherical_harmonics import SphericalHarmonicProjection, InverseSphericalHarmonicProjection
from get_directions import get_directions
from loaders import (
    load_config,
    load_datamodule,
    load_logger,
    load_model,
)

torch.set_float32_matmul_precision("medium")


def beta_schedule(epoch: int, period: int, beta_min: float, beta_max: float) -> float:
    """Cosine ramp from beta_min → beta_max over `period` epochs, then holds."""
    if period <= 0:
        return beta_max
    t = min(epoch / period, 1.0)
    return beta_min + 0.5 * (beta_max - beta_min) * (
        1 - torch.cos(torch.tensor(t * 3.14159)).item()
    )


def weighted_sh_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    l_max: int,
) -> torch.Tensor:
    loss   = torch.tensor(0.0, device=pred.device)
    offset = 0
    for l in range(l_max + 1):
        m    = 2 * l + 1
        w    = 1.0 / m
        loss = loss + w * torch.nn.functional.mse_loss(
            pred[...,   offset:offset + m],
            target[..., offset:offset + m],
        )
        offset += m
    return loss


def compute_ect_chunked(
    x: torch.Tensor,
    v: torch.Tensor,
    radius: float,
    resolution: int,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    num_dirs = v.shape[1]
    lin      = torch.linspace(-radius, radius, resolution, device=x.device)  
    chunks   = []

    for start in range(0, num_dirs, chunk_size):
        v_chunk = v[:, start:start + chunk_size]        
        nh      = x @ v_chunk                            
        ecc     = torch.sigmoid(
            scale * (lin.view(1, 1, 1, resolution) - nh.unsqueeze(-1))
        )                                                 
        ect_chunk = ecc.sum(dim=1)                
        chunks.append(ect_chunk)

    ect = torch.cat(chunks, dim=1) 
    ect = 2 * (ect / ect.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)) - 1
    return ect


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, max_points: int = 256) -> torch.Tensor:
    B, N, D = pc1.shape
    _, M, _ = pc2.shape
    assert N == M, "Chamfer in ECT space assumes same num_dirs in both tensors"

    if N > max_points:
        idx = torch.randperm(N, device=pc1.device)[:max_points]
        pc1 = pc1[:, idx, :]
        pc2 = pc2[:, idx, :]

    diff  = pc1.unsqueeze(2) - pc2.unsqueeze(1)  
    dist2 = (diff ** 2).sum(dim=-1)           
    cd    = dist2.min(dim=2).values.mean(dim=1) \
          + dist2.min(dim=1).values.mean(dim=1) 
    return cd.mean()


def train(
    trainerconfig,
    modelconfig,
    loggerconfig,
    fabric,
    dataloader,
    valdataloader,
    model,
    optimizer_g,
    scheduler,
    logger,
    dirs,                
    sh_transform,      
    inverse_sh_transform,  
    no_progressbar,
):
    """
    pc → ECT → SH  → VAE → SH_recon → ECT_recon

    Loss = KLD + MSE(SH, SH_recon) + Chamfer(ECT, ECT_recon)
    """
    ect_cfg    = modelconfig.ectconfig
    v          = dirs.T.to(fabric.device)  
    chunk_size = getattr(ect_cfg, "chunk_size", 64) 

    step_count = 0
    for epoch in range(trainerconfig.max_epochs):
        model.train()

        for pc in tqdm(dataloader, disable=no_progressbar):
            step_count += 1
            optimizer_g.zero_grad()

            # Point Cloud → ECT 
            with torch.no_grad():
                ect = compute_ect_chunked(
                    x          = pc,
                    v          = v,
                    radius     = ect_cfg.r,
                    resolution = ect_cfg.resolution,
                    scale      = ect_cfg.scale,
                    chunk_size = chunk_size,
                )   

                # ECT → SH
                sh_gt = sh_transform(ect)   

            # VAE forward
            sh_recon, mu, logvar = model(sh_gt)  

            # SH → ECT reconstruction
            ect_recon = inverse_sh_transform(sh_recon)  

            # KLD with free bits — prevents posterior collapse on low-info dims.
            logvar_exp  = model.expand_logvar(logvar.clamp(min=-10, max=10))
            kld_per_dim = -0.5 * (1 + logvar_exp - mu ** 2 - logvar_exp.exp())
            FREE_BITS   = 0.5  
            kld_loss    = torch.clamp(kld_per_dim, min=FREE_BITS).sum(dim=1).mean()

            mse_loss = weighted_sh_mse(sh_recon, sh_gt, model.l_max)

            cd_loss = chamfer_distance(ect, ect_recon)

            beta   = beta_schedule(
                epoch,
                modelconfig.beta_period,
                modelconfig.beta_min,
                modelconfig.beta_max,
            )
            g_loss = mse_loss + beta * kld_loss + cd_loss

            fabric.backward(g_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            logger.log_metrics(
                {
                    "g_loss":   g_loss.item(),
                    "kld_loss": kld_loss.item(),
                    "mse_loss": mse_loss.item(),
                    "cd_loss":  cd_loss.item(),
                    "beta":     beta,
                },
                step=step_count,
            )
            if step_count % 10 == 0:  
                print(
                    f"Epoch {epoch:03d} | Step {step_count:06d} | "
                    f"Total Loss: {g_loss.item():.4f} | "
                    f"MSE: {mse_loss.item():.4f} | "
                    f"CD: {cd_loss.item():.4f} | "
                    f"KLD: {kld_loss.item():.4f} | "
                    f"beta: {beta:.8f}"
                )

        if epoch % trainerconfig.checkpoint_interval == 0:
            state     = {"model": model}
            ckpt_path = f"results/{loggerconfig.results_dir}/model.ckpt"
            print(f"Saving model to {ckpt_path}")
            fabric.save(ckpt_path, state)

            _save_sh_comparison(
                sh_gt, sh_recon, epoch,
                path=f"results/{loggerconfig.results_dir}/sh_recon_{epoch:04}.png",
            )

        scheduler.step()
        logger.log_metrics({"lr": scheduler.get_last_lr()[0]}, step=step_count)

    state      = {"model": model}
    final_path = f"results/{loggerconfig.results_dir}/model.ckpt"
    print(f"Final save to {final_path}")
    fabric.save(final_path, state)

    os.makedirs(f"results/{loggerconfig.results_dir}/test", exist_ok=True)
    gt_pcs_list = []
    recon_sh_all, gt_sh_all, recon_ect_all, gt_ect_all = [], [], [], []
    model.eval()

    with torch.no_grad():
        for idx, pc in enumerate(valdataloader):
            ect = compute_ect_chunked(
                x=pc, v=v,
                radius=ect_cfg.r,
                resolution=ect_cfg.resolution,
                scale=ect_cfg.scale,
                chunk_size=chunk_size,
            )
            sh_gt     = sh_transform(ect)
            sh_recon, _, _ = model(sh_gt)
            ect_recon = inverse_sh_transform(sh_recon)

            # Sample from prior
            sh_sample_flat = model.sample(len(pc), device=fabric.device)
            resolution_    = sh_gt.shape[1]
            sh_dim_        = sh_gt.shape[2]
            sh_sample      = sh_sample_flat.reshape(len(pc), resolution_, sh_dim_)
            _ect_sample    = inverse_sh_transform(sh_sample) 

            recon_sh_all.append(sh_recon.cpu())
            gt_sh_all.append(sh_gt.cpu())
            recon_ect_all.append(ect_recon.cpu())
            gt_ect_all.append(ect.cpu())
            gt_pcs_list.append(pc.cpu()) 

            if idx == 0:
                _save_sh_comparison(
                    sh_gt, sh_recon,
                    epoch=trainerconfig.max_epochs - 1,
                    path=f"results/{loggerconfig.results_dir}/sh_recon_test.png",
                )

    torch.save(torch.vstack(recon_sh_all),  f"results/{loggerconfig.results_dir}/recon_sh.pt")
    torch.save(torch.vstack(gt_sh_all),     f"results/{loggerconfig.results_dir}/gt_sh.pt")
    torch.save(torch.vstack(recon_ect_all), f"results/{loggerconfig.results_dir}/recon_ect.pt")
    torch.save(torch.vstack(gt_ect_all),    f"results/{loggerconfig.results_dir}/gt_ect.pt") 
    torch.save(torch.vstack(gt_pcs_list), f"results/{loggerconfig.results_dir}/gt_pcs.pt")

def _save_sh_comparison(sh_gt, sh_recon, epoch, path):
    if sh_gt.dim() < 2:
        return
    try:
        n = min(8, sh_gt.shape[0])

        def to_strip(t):
            t = t[:n].detach().cpu().float()
            t = t.reshape(n, 1, 1, -1)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            return t

        strips = torch.cat([to_strip(sh_gt), to_strip(sh_recon)], dim=0)
        grid   = make_grid(strips, nrow=n)
        img    = torchvision.transforms.ToPILImage()(grid)
        img.save(path)
        img.close()
    except Exception as e:
        print(f"[warn] Could not save SH comparison at epoch {epoch}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train equivariant SH-VAE")
    parser.add_argument("--config",         dest="config_path",
                        default="configs/equiv_vae.yaml", type=str)
    parser.add_argument("--compile",        default=False, action="store_true")
    parser.add_argument("--dev",            default=False, action="store_true")
    parser.add_argument("--resume",         default=False, action="store_true")
    parser.add_argument("--no-progressbar", default=False, action="store_true")
    args = parser.parse_args()

    (
        dataconfig,
        _transformconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    ) = load_config(args.config_path)

    if args.dev:
        trainerconfig.max_epochs = 10

    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )
    seed_everything(trainerconfig.seed)

    logger = load_logger(loggerconfig)

    dm = load_datamodule(dataconfig, dev=args.dev)
    dataloader    = fabric.setup_dataloaders(dm.train_dataloader)
    valdataloader = fabric.setup_dataloaders(dm.val_dataloader)

    dirs, weights = get_directions(num_points=modelconfig.ectconfig.num_thetas)

    assert modelconfig.sh_input_channels == modelconfig.ectconfig.resolution

    sh_transform = SphericalHarmonicProjection(
        dirs=dirs,
        weights=weights,
        l_max=modelconfig.l_max,
    )
    sh_transform = fabric.setup_module(sh_transform)

    inverse_sh_transform = InverseSphericalHarmonicProjection(
        dirs=dirs,
        l_max=modelconfig.l_max,
    )
    inverse_sh_transform = fabric.setup_module(inverse_sh_transform)

    model = load_model(modelconfig)

    if args.resume:
        ckpt = f"results/{loggerconfig.results_dir}/model.ckpt"
        print(f"Resuming from {ckpt}")
        fabric.load(ckpt, {"model": model})

    if args.compile:
        model = torch.compile(model)

    optimizer_g = Adam(
        model.parameters(),
        lr=modelconfig.lr,
        betas=(0.5, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=trainerconfig.max_epochs,
        eta_min=1e-6,
    )
    model, optimizer_g = fabric.setup(model, optimizer_g)

    train(
        trainerconfig,
        modelconfig,
        loggerconfig,
        fabric,
        dataloader,
        valdataloader,
        model,
        optimizer_g,
        scheduler,
        logger,
        dirs,
        sh_transform,
        inverse_sh_transform,
        args.no_progressbar,
    )


if __name__ == "__main__":
    main()
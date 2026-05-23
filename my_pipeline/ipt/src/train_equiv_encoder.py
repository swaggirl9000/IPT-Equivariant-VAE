import argparse
import os

import lightning
import pydantic
import torch
import torch.nn.functional as F
import torchvision
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid
from tqdm import tqdm
from layers.ect import EctConfig

from loaders import (
    load_config,
    load_datamodule,
    load_logger,
    load_model,
    load_transform,
)
from metrics.loss import chamfer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_recon_3d(pcs_recon, pcs_gt, num_pc=8, filename="plot.png"):
    recon  = pcs_recon.detach().cpu().numpy()
    gt     = pcs_gt.detach().cpu().numpy()
    n_plot = min(num_pc, len(recon), len(gt))

    fig = plt.figure(figsize=(4 * n_plot, 8))
    for i in range(n_plot):
        ax_gt = fig.add_subplot(2, n_plot, i + 1, projection='3d')
        ax_gt.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2], c='#1f77b4', s=3, alpha=0.6)
        ax_gt.set_title(f"GT {i}")
        ax_gt.axis('off')

        ax_rec = fig.add_subplot(2, n_plot, n_plot + i + 1, projection='3d')
        ax_rec.scatter(recon[i, :, 0], recon[i, :, 1], recon[i, :, 2], c='#d62728', s=3, alpha=0.6)
        ax_rec.set_title(f"Recon {i}")
        ax_rec.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


from transforms.ecttransform import Transform, TransformConfig
from spherical_harmonics import SphericalHarmonicProjection

torch.set_float32_matmul_precision("medium")

# def compute_ipt(
#     pc: torch.Tensor,      
#     dirs: torch.Tensor,     
#     num_thetas: int,        
#     r: float,
# ) -> torch.Tensor:          
#     projections = pc @ dirs.T                                     
#     thetas = torch.linspace(-r, r, num_thetas, device=pc.device)  
#     proj   = projections.unsqueeze(-1)                     
#     t      = thetas.reshape(1, 1, 1, num_thetas)               
#     ect    = torch.sigmoid(20.0 * (t - proj)).sum(dim=1)          
#     return ect

def compute_ipt(pc, dirs, resolution, r):
    projections = pc @ dirs.T
    thetas = torch.linspace(-r, r, resolution, device=pc.device)  # resolution, not num_thetas
    proj   = projections.unsqueeze(-1)
    t      = thetas.reshape(1, 1, 1, resolution)
    ect    = torch.sigmoid(20.0 * (t - proj)).sum(dim=1)
    return ect  


def train(
    fabric,
    dataloader,
    transform,
    model,
    optimizer,
    losstransform,
    dirs,
    resolution,
    r,
    sh_projection,
    trainerconfig,
    loggerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
    m,
    s,
):
    model.train()
    step = 0
    for epoch in range(trainerconfig.max_epochs):
        batch_idx = -1
        for pcs in tqdm(dataloader, disable=no_progressbar):
            optimizer.zero_grad(set_to_none=True)
            batch_idx += 1
            if transform is not None:
                pcs = transform(pcs)

            # Point Cloud → IPT → SH → Equivariant Decoder
            ipt_input   = compute_ipt(pcs, dirs, resolution, r)  
            sh_features = sh_projection(ipt_input)               
            pcs_recon   = model(sh_features)                    

            # Loss: MSE on IPT space + Chamfer on point cloud space
            ipt_gt    = losstransform(pcs)
            ipt_recon = losstransform(pcs_recon)

            ipt_loss = F.mse_loss(ipt_gt, ipt_recon)
            cd_loss  = chamfer(pcs_recon, pcs)
            loss     = cd_loss + 10.0 * ipt_loss

            fabric.backward(loss)
            optimizer.step()

            logger.log_metrics(
                {"loss": loss, "ipt_loss": ipt_loss, "cd_loss": cd_loss},
                step=step,
            )
            step += 1

            if batch_idx == 0 and epoch % trainerconfig.checkpoint_interval == 0:
                state = {"model": model}
                if not dev:
                    print(f" Saving model to {results_base_dir}/model.ckpt")
                    fabric.save(f"{results_base_dir}/model.ckpt", state)
                print(f"Saving screenshot to: {results_base_dir}/pcs_recon_{epoch:04}.png")
                plot_recon_3d(
                    pcs_recon[:8], pcs[:8], num_pc=8,
                    filename=f"{results_base_dir}/pcs_recon_{epoch:04}.png",
                )

    state = {"model": model}
    fabric.save(f"{results_base_dir}/model.ckpt", state)


@torch.no_grad()
def test(
    fabric,
    dataloader,
    transform,
    model,
    optimizer,
    losstransform,
    dirs,
    resolution,
    r,
    sh_projection,
    trainerconfig,
    loggerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
    m,
    s,
):
    model.eval()
    pcs_recon_list = []
    pcs_gt_list    = []

    for pcs in tqdm(dataloader, disable=no_progressbar):
        # Point Cloud → IPT → SH → Equivariant Decoder
        ipt_input = compute_ipt(pcs, dirs, resolution, r)
        sh_features = sh_projection(ipt_input)
        pcs_recon   = model(sh_features)

        ipt_gt    = losstransform(pcs)
        ipt_recon = losstransform(pcs_recon)

        ipt_loss = F.mse_loss(ipt_gt, ipt_recon)
        cd_loss  = chamfer(pcs_recon, pcs)

        pcs_recon_list.append(pcs_recon.detach().cpu())
        pcs_gt_list.append(pcs.detach().cpu())

    pcs_recon = torch.vstack(pcs_recon_list).cpu()
    pcs_gt    = torch.vstack(pcs_gt_list).cpu()

    if m is not None and s is not None:
        pcs_recon = pcs_recon * s.cpu() + m.cpu()
        pcs_gt    = pcs_gt    * s.cpu() + m.cpu()

    torch.save(pcs_recon, f"{results_base_dir}/pcs_recon.pt")
    torch.save(pcs_gt,    f"{results_base_dir}/pcs_gt.pt")
    print(f"Saving screenshot to: {results_base_dir}/pcs_final.png")
    plot_recon_3d(
        pcs_recon_list[0], pcs_gt_list[0], num_pc=8,
        filename=f"{results_base_dir}/pcs_final.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Arguments for encoder training")
    parser.add_argument("--config",  dest="config_path",
                        default="configs/encoder_airplane.yaml", type=str)
    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--dev",     default=False, action="store_true")
    parser.add_argument("--resume",  default=False, action="store_true")
    parser.add_argument("--no-progressbar", default=False, action="store_true")
    args = parser.parse_args()

    (
        dataconfig, transformconfig, modelconfig,
        trainerconfig, loggerconfig,
    ) = load_config(args.config_path)

    if isinstance(modelconfig.ectconfig, dict):
        modelconfig.ectconfig = EctConfig(**modelconfig.ectconfig)
    if isinstance(modelconfig.ectlossconfig, dict):
        modelconfig.ectlossconfig = EctConfig(**modelconfig.ectlossconfig)
    
    results_base_dir = "results"
    if args.dev:
        trainerconfig.max_epochs = 5000
        results_base_dir += "_dev"
    results_base_dir += f"/{loggerconfig.results_dir}"
    os.makedirs(results_base_dir, exist_ok=True)

    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )
    seed_everything(trainerconfig.seed)

    logger = load_logger(loggerconfig)

    dm = load_datamodule(dataconfig, dev=args.dev)
    m, s = dm.m, dm.s
    dataloader    = fabric.setup_dataloaders(dm.train_dataloader)
    valdataloader = fabric.setup_dataloaders(dm.val_dataloader)

    if transformconfig is not None:
        transform = load_transform(transformconfig)
        transform = fabric.setup_module(transform)
    else:
        transform = None

    model = load_model(modelconfig)
    print(summary(model))

    if args.resume:
        print(f"Resuming from: {results_base_dir}/model.ckpt")
        state = {"model": model}
        fabric.load(f"{results_base_dir}/model.ckpt", state)

    if args.compile:
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=modelconfig.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    ect_transform_config = TransformConfig(module="", ectconfig=modelconfig.ectconfig)
    ect_transform_layer  = Transform(ect_transform_config)

    v_raw = ect_transform_layer.v                         
    dirs  = (v_raw.T if v_raw.shape[0] == 3 else v_raw).to(fabric.device)  
    assert dirs.shape[1] == 3, f"Expected [num_dirs, 3], got {dirs.shape}"

    resolution = modelconfig.ectconfig.resolution
    r          = float(modelconfig.ectconfig.r)
    print(f"IPT: num_dirs={dirs.shape[0]}, resolution={resolution}, r={r}")

    weights       = torch.ones(dirs.shape[0], device=fabric.device)
    sh_projection = SphericalHarmonicProjection(
        dirs=dirs, weights=weights, l_max=modelconfig.lmax
    )
    sh_projection = fabric.setup_module(sh_projection)

    loss_transform_config = TransformConfig(module="", ectconfig=modelconfig.ectlossconfig)
    losstransform = Transform(loss_transform_config)
    losstransform = fabric.setup_module(losstransform)

    shared = dict(
        dirs=dirs, resolution=resolution, r=r,
        sh_projection=sh_projection,
        trainerconfig=trainerconfig, loggerconfig=loggerconfig,
        logger=logger, no_progressbar=args.no_progressbar,
        results_base_dir=results_base_dir, dev=args.dev, m=m, s=s,
    )

    train(fabric, dataloader,    transform, model, optimizer, losstransform, **shared)
    test( fabric, valdataloader, transform, model, optimizer, losstransform, **shared)


if __name__ == "__main__":
    main()
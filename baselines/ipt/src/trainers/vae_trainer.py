import os
import matplotlib.pyplot as plt

import lightning as L
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid
from plotting import ECT_PLOT_CONFIG

from models.vae_baseline import Model
from transforms.ecttransform import Transform, TransformConfig


class VAETrainer(L.LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        self._last_val_batch = None

        self.model = Model(latent_dim=cfg.modelconfig.latent_dim, input_resolution=cfg.modelconfig.ectconfig.resolution)
        self.loss_fn_recon = torch.nn.MSELoss()

        ect_transform_cfg = TransformConfig(
            module="", ectconfig=cfg.modelconfig.ectconfig
        )
        self.ecttransform = Transform(ect_transform_cfg)

        self.test_outputs = {}

    def _beta(self) -> float:
        period = self.cfg.modelconfig.beta_period
        step = self.global_step % period
        frac = step / period
        beta = self.cfg.modelconfig.beta_min + (
            self.cfg.modelconfig.beta_max - self.cfg.modelconfig.beta_min
        ) * frac
        return float(beta)

    @staticmethod
    def _kld_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        )

    def _is_wandb(self) -> bool:
        return isinstance(self.logger, L.pytorch.loggers.WandbLogger)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        pc = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Compute ECT on-the-fly from raw point cloud
        ect = self.ecttransform(pc).unsqueeze(1)

        output, mu, log_var = self.model(ect)

        recon_loss = self.loss_fn_recon(output, ect)
        kld_loss = self._kld_loss(mu, log_var)
        beta = self._beta()
        total_loss = recon_loss + beta * kld_loss

        self.log_dict(
            {
                f"{stage}/loss": total_loss,
                f"{stage}/recon_loss": recon_loss,
                f"{stage}/kld_loss": kld_loss,
                f"{stage}/beta": beta,
            },
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss, output, ect

    def training_step(self, batch, batch_idx: int):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, output, ect = self._shared_step(batch, "val")

        if batch_idx == 0 and self._is_wandb():
            n = min(8, ect.size(0))
            save_recon = (1 + torch.clamp(output[:n], -1.0, 1.0)) / 2
            save_gt = (ect[:n] + 1) / 2
            grid = make_grid(torch.cat([save_gt, save_recon], dim=0), nrow=n)
            self.logger.experiment.log(
                {
                    "val/reconstructions": wandb.Image(
                        grid.permute(1, 2, 0).cpu().numpy(),
                        caption="Top: Ground Truth | Bottom: Reconstruction",
                    )
                }
            )
        
        if batch_idx == 0:
            self._last_val_batch = (ect.detach().cpu(), output.detach().cpu())
            
        return loss

    def on_test_start(self):
        self.test_outputs = {"recon": [], "gt": [], "samples": []}

    def test_step(self, batch, batch_idx: int):
        pc = batch[0] if isinstance(batch, (list, tuple)) else batch
        ect = self.ecttransform(pc).unsqueeze(1)

        output, _, _ = self.model(ect)
        sample = self.model.sample(len(pc), device=self.device)

        self.test_outputs["recon"].append(output.cpu())
        self.test_outputs["gt"].append(ect.cpu())
        self.test_outputs["samples"].append(sample.cpu())

    def on_test_end(self):
        res_dir = f"results/{self.cfg.logger.results_dir}/test"
        os.makedirs(res_dir, exist_ok=True)
        torch.save(torch.vstack(self.test_outputs["recon"]), f"{res_dir}/recon_ect.pt")
        torch.save(torch.vstack(self.test_outputs["gt"]),   f"{res_dir}/gt_ect.pt")
        torch.save(torch.vstack(self.test_outputs["samples"]), f"{res_dir}/sample_ect.pt")
        print(f"Test outputs saved to {res_dir}/")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.modelconfig.learning_rate,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.cfg.trainer.max_epochs,
            eta_min=self.cfg.modelconfig.learning_rate * 1e-2,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
    
    def save_plots(self, plot_dir: str, epoch: int):
        if self._last_val_batch is None:
            return

        os.makedirs(plot_dir, exist_ok=True)
        gt, recon = self._last_val_batch
        n = min(5, gt.size(0))

        fig, axes = plt.subplots(nrows=3, ncols=n, figsize=(3 * n, 9))
        for i in range(n):
            gt_img    = gt[i].squeeze().numpy()
            recon_img = recon[i].squeeze().numpy()
            error_img = abs(gt_img - recon_img)

            axes[0, i].imshow(gt_img,    **ECT_PLOT_CONFIG); axes[0, i].axis("off")
            axes[1, i].imshow(recon_img, **ECT_PLOT_CONFIG); axes[1, i].axis("off")
            axes[2, i].imshow(error_img, cmap="hot", vmin=0, vmax=1); axes[2, i].axis("off")

        axes[0, 0].set_ylabel("Original",       fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed",  fontsize=12)
        axes[2, 0].set_ylabel("Error |gt-recon|", fontsize=12)

        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"recon_{epoch:04d}.png"), dpi=150)
        plt.close("all")

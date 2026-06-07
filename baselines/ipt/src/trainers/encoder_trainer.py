import os

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf

from metrics.loss import chamfer
from models.encoder import Model, ModelConfig
from transforms.ecttransform import Transform, TransformConfig


class EncoderTrainer(L.LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        model_cfg = ModelConfig(**OmegaConf.to_container(cfg.modelconfig, resolve=True))
        self.model = Model(model_cfg)

        ect_transform_cfg = TransformConfig(
            module="", ectconfig=model_cfg.ectconfig
        )
        loss_transform_cfg = TransformConfig(
            module="", ectconfig=model_cfg.ectlossconfig
        )
        self.ecttransform = Transform(ect_transform_cfg)
        self.losstransform = Transform(loss_transform_cfg)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        ect = self.ecttransform(pc).unsqueeze(1)
        return self.model(ect)

    def _shared_step(self, batch, stage: str):
        pcs = batch[0] if isinstance(batch, (list, tuple)) else batch

        ect = self.ecttransform(pcs).unsqueeze(1)
        pcs_recon = self.model(ect)

        ect_gt   = self.losstransform(pcs)
        ect_recon = self.losstransform(pcs_recon)
        ect_loss = F.mse_loss(ect_gt, ect_recon)
        cd_loss  = chamfer(pcs_recon, pcs)
        loss     = cd_loss + 10 * ect_loss

        self.log_dict(
            {
                f"{stage}/loss":     loss,
                f"{stage}/ect_loss": ect_loss,
                f"{stage}/cd_loss":  cd_loss,
            },
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss, pcs_recon, pcs

    def training_step(self, batch, batch_idx: int):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, pcs_recon, pcs_gt = self._shared_step(batch, "val")

        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            n = min(4, pcs_recon.size(0))
            self.logger.experiment.log(
                {
                    "val/pred_pointcloud": wandb.Object3D(
                        pcs_recon[:n].detach().cpu().numpy().reshape(-1, 3)
                    ),
                    "val/gt_pointcloud": wandb.Object3D(
                        pcs_gt[:n].detach().cpu().numpy().reshape(-1, 3)
                    ),
                }
            )
        return loss

    def on_test_start(self):
        self.test_outputs = {"pred_pc": [], "gt_pc": []}

    def test_step(self, batch, batch_idx: int):
        pcs = batch[0] if isinstance(batch, (list, tuple)) else batch

        ect = self.ecttransform(pcs).unsqueeze(1)
        pcs_recon = self.model(ect)

        self.test_outputs["pred_pc"].append(pcs_recon.cpu())
        self.test_outputs["gt_pc"].append(pcs.cpu())

    def on_test_end(self):
        res_dir = f"results/{self.cfg.logger.results_dir}/test"
        os.makedirs(res_dir, exist_ok=True)

        pcs_recon = torch.vstack(self.test_outputs["pred_pc"])
        pcs_gt    = torch.vstack(self.test_outputs["gt_pc"])

        dm = self.trainer.datamodule
        if hasattr(dm, "m") and hasattr(dm, "s") and dm.m is not None:
            m, s = dm.m.cpu(), dm.s.cpu()
            pcs_recon = pcs_recon * s + m
            pcs_gt    = pcs_gt    * s + m

        torch.save(pcs_recon, f"{res_dir}/pcs_recon.pt")
        torch.save(pcs_gt,    f"{res_dir}/pcs_gt.pt")
        print(f"Test outputs saved to {res_dir}/")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.modelconfig.learning_rate,
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
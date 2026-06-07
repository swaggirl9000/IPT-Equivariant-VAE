"""
train.py — unified Hydra entry point for all models.

Usage
-----
# Train the VAE (default)
python train.py

# Train the encoder
python train.py modelconfig@modelconfig=encoder logger.results_dir=encoder_run

# Dev smoke-test
python train.py dev=true

# Resume from checkpoint
python train.py +resume=checkpoints/vae_baseline/last.ckpt

# Evaluate after training
python train.py evaluate=true

# Sweep
python train.py --multirun modelconfig.latent_dim=64,128,256
"""

import importlib
import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from trainers.plot_callback import PlotCallback


from loaders import load_datamodule, load_logger

torch.set_float32_matmul_precision("medium")

TRAINER_REGISTRY = {
    "models.vae_baseline": "trainers.vae_trainer.VAETrainer",
    "models.encoder":      "trainers.encoder_trainer.EncoderTrainer",
}


def _load_trainer_class(module_key: str):
    """Resolve a dotted class path like 'models.vae_trainer.VAETrainer'."""
    trainer_path = TRAINER_REGISTRY.get(module_key)
    if trainer_path is None:
        raise ValueError(
            f"No trainer registered for modelconfig.module='{module_key}'. "
            f"Known keys: {list(TRAINER_REGISTRY)}"
        )
    module_path, class_name = trainer_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    if cfg.get("dev", False):
        cfg.trainer.max_epochs = 2
        cfg.trainer.save_dir = cfg.trainer.save_dir + "_dev"
        cfg.logger.results_dir = cfg.logger.results_dir + "_dev"
        if hasattr(cfg.logger, "tags"):
            cfg.logger.tags.append("dev")

    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.trainer.seed, workers=True)

    ckpt_dir = os.path.join(cfg.trainer.save_dir, cfg.logger.results_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    dm = load_datamodule(cfg.data, dev=cfg.get("dev", False))

    TrainerClass = _load_trainer_class(cfg.modelconfig.module)
    resume_path: str | None = cfg.get("resume", None)

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        model = TrainerClass.load_from_checkpoint(resume_path, cfg=cfg)
    else:
        model = TrainerClass(cfg)

    if cfg.get("compile", False):
        model = torch.compile(model)

    logger = load_logger(cfg.logger)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:04d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=cfg.trainer.get("save_top_k", 3),
        ),
        PlotCallback(                                   
        every_n_epochs=cfg.trainer.get("plot_every_n_epochs", -1),
        plot_dir=cfg.trainer.get("plot_dir", "plots"),
    ),
    ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=[PlotCallback(every_n_epochs=cfg.trainer.get("plot_every_n_epochs", -1), 
                                plot_dir=ckpt_dir)] + callbacks,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.get("devices", "auto"),
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 10),
        precision=cfg.trainer.get("precision", "32-true") or "32-true",
        enable_progress_bar=True,
        deterministic=cfg.get("deterministic", False),
    )

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=resume_path,
    )
    if cfg.get("evaluate", False):
        trainer.test(model, dataloaders=dm.test_dataloader)


if __name__ == "__main__":
    main()

"""
ploit for evaluatiopn after few epochs, 
disable in config, plot ever n eppchs where -1 is never 
create intenralfile to test if IPT computatiopn is correct, by plotting original, reocnstructed, and error images.
this is for the encoder 


""" 


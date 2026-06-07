import functools
import importlib
import json
import timeit
from types import SimpleNamespace
from typing import Any

import pydantic
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn

from shapenet_datamodule import ShapeNetDataModule

DATAMODULE_REGISTRY = {
    "shapenet": ShapeNetDataModule,
}

def load_module(config_dict: dict[Any, Any], classname: str) -> pydantic.BaseModel:

    module_name = config_dict.get("module", None)

    if module_name is not None:
        module = importlib.import_module(config_dict["module"])
        config_class = getattr(module, classname)
        config = config_class(**config_dict)
    else:
        module = importlib.import_module("__main__")
        config_class = getattr(module, classname)
        config = config_class(**config_dict)
    return config


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict: dict[str, Any] = yaml.safe_load(stream)

    # Data
    dataconfig = load_module(config_dict["data"], classname="DataConfig")

    # Transform
    transform_list = config_dict.get("transform", None)
    transformconfig = None
    if transform_list is not None:
        transformconfig = [
            load_module(cfg, classname="TransformConfig")
            for cfg in config_dict["transform"]
        ]

    # Model
    modelconfig = load_module(config_dict["modelconfig"], classname="ModelConfig")

    # Trainer
    trainerconfig = load_module(config_dict["trainer"], classname="TrainerConfig")

    # Logger
    loggerconfig = load_module(config_dict["logger"], classname="LogConfig")

    return dataconfig, transformconfig, modelconfig, trainerconfig, loggerconfig


def load_datamodule(config, dev: bool = False):
    if config.module in DATAMODULE_REGISTRY:
        return DATAMODULE_REGISTRY[config.module](config)

    module = importlib.import_module(config.module)
    train_dl, val_dl, test_dl, m, s = module.get_all_dataloaders(config, dev=dev)
    return SimpleNamespace(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        m=m,
        s=s,
    )


def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "Model")

    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        config_dict = json.loads(json.dumps(config, default=lambda s: vars(s)))
    model = model_class(config)
    return model


def load_transform(config):
    transforms = []
    for tr_config in config:
        module = importlib.import_module(tr_config.module)
        transform_class = getattr(module, "Transform")
        transform = transform_class(tr_config)
        transforms.append(transform)

    return nn.Sequential(*transforms)


def load_object(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    else:
        return obj


def load_logger(config):
    """
    Loads the logger.
    """
    module = importlib.import_module(config.module)
    return module.load_logger(config)

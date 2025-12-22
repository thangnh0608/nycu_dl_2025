from typing import Any, Dict
import os
import argparse
import logging
import csv
import numpy as np
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from omegaconf import OmegaConf
import mlflow


def make_optimizer_with_layerwise_lr(model, cfg):
    base_lr = cfg.training.lr
    w_decay = cfg.training.w_decay
    layerwise_lr_decay = cfg.training.get('layerwise_lr_decay', 0.1)
    if hasattr(model, "classifier"):
        head_params = list(model.classifier.parameters())
        head_param_ids = {id(p) for p in head_params}
    elif hasattr(model, "fc"):
        head_params = list(model.fc.parameters())
        head_param_ids = {id(p) for p in head_params}
    else:
        raise ValueError("Cannot find classifier head in model.")
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
            or "layernorm" in name.lower()
            or "ln" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Split head/body again inside decay categories
    head_decay       = [p for p in decay_params if id(p) in head_param_ids]
    body_decay       = [p for p in decay_params if id(p) not in head_param_ids]
    head_no_decay    = [p for p in no_decay_params if id(p) in head_param_ids]
    body_no_decay    = [p for p in no_decay_params if id(p) not in head_param_ids]

    # --------------------
    # 4. Build parameter groups
    # --------------------
    param_groups = [
        {"params": body_decay,    "lr": base_lr * layerwise_lr_decay, "weight_decay": w_decay},
        {"params": body_no_decay, "lr": base_lr * layerwise_lr_decay, "weight_decay": 0.0},
        {"params": head_decay,    "lr": base_lr,       "weight_decay": w_decay},
        {"params": head_no_decay, "lr": base_lr,       "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_experiment_artifacts(results, save_path, logger=None, suffix=""):
    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(save_path, f"test_results_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(results)

    if logger is not None:
        logger.info(f"Saved test results to {csv_path}")
        mlflow.log_artifact(csv_path)

        for h in logger.handlers:
            if hasattr(h, "flush"):
                h.flush()

        mlflow.log_artifact(os.path.join(save_path, "train.log"))


def flatten_omegaconf(cfg, resolve: bool = True) -> Dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=resolve)

    def _flatten(d: Dict, parent_key: str = "") -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    return _flatten(container)

if __name__ == "__main__":
    test_cfg = OmegaConf.load("config.yaml")
    a = flatten_omegaconf(test_cfg, resolve=True)
    print(a)
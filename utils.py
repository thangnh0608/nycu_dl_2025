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
from torchvision.transforms import v2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from omegaconf import OmegaConf
import mlflow

def apply_overrides(obj, overrides: dict):
    """Recursively apply nested dict overrides to a config object."""
    for k, v in overrides.items():
        if not hasattr(obj, k):
            raise KeyError(f"Invalid config key: {k}")
        attr = getattr(obj, k)
        # If value is a dict and attribute is a nested config
        if isinstance(v, dict) and hasattr(attr, "__dict__"):
            apply_overrides(attr, v)
        else:
            setattr(obj, k, v)

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

    csv_submit_path = os.path.join(save_path, f"submission_{suffix}.csv")
    csv_score_path  = os.path.join(save_path, f"test_results_{suffix}.csv")


    with open(csv_submit_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows([(fn, lb) for fn, lb, _ in results])

    with open(csv_score_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        writer.writerows(results)

    mlflow.log_artifact(csv_score_path)

    if logger is not None:
        logger.info(f"Saved test results to {csv_score_path}")

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


def get_transforms_from_cfg(config_dict, processor=None):
    augmen_config = config_dict.augmentation
    transforms_list = [
        v2.ToImage(),
        v2.Resize(config_dict.dataset.image_size)
    ]
    late_transforms = []
    if augmen_config is not None:
        augmen_config = OmegaConf.to_container(augmen_config, resolve=True)
    else:
        augmen_config = {}

    for transform_name, params in augmen_config.items():
        transform_class = getattr(v2, transform_name)
        if params is not None:
            instance = transform_class(**params) if isinstance(params, dict) else transform_class(params)
        else:
            instance = transform_class()

        if transform_name == "RandomErasing":
            late_transforms.append(instance)
        else:
            transforms_list.append(instance)

    # 2. Append ToTensor
    transforms_list.append(v2.ToDtype(torch.float32, scale=True),)
    transforms_list.extend(late_transforms)
    # 3. Append Normalize with HF processor mean/std if available
    if processor and hasattr(processor, "image_mean") and hasattr(processor, "image_std"):
        mean = processor.image_mean
        std = processor.image_std
    else:
        try:
            mean = processor.mean
            std = processor.std
        except:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    transforms_list.append(v2.Normalize(mean=mean, std=std))
    transform = v2.Compose(transforms_list)
    eval_transform = transforms.Compose([
        v2.ToImage(),
        v2.Resize((config_dict.dataset.image_size, config_dict.dataset.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    return transform, eval_transform

def apply_overrides(obj, overrides: dict):
    """Recursively apply nested dict overrides to a config object."""
    for k, v in overrides.items():
        if not hasattr(obj, k):
            raise KeyError(f"Invalid config key: {k}")
            
        attr = getattr(obj, k)
        # If value is a dict and attribute is a nested config
        if isinstance(v, dict) and hasattr(attr, "__dict__"):
            apply_overrides(attr, v)
        else:
            setattr(obj, k, v)

def run_inference(device, model, ema, class_to_idx, test_dl):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, filenames in test_dl:
            imgs = imgs.to(device)
            # batch = processor(images=imgs, return_tensors="pt", ).to(device)
            outputs = ema.module(imgs)
            score = torch.softmax(outputs.logits, dim=1).cpu().tolist()
            preds = outputs.logits.argmax(1).cpu().tolist()

            for f, p, s in zip(filenames, preds, score):
                results.append((f[:-4], class_to_idx[p], s))
    return results

def make_transforms(cfg, processor):
    transform, eval_transform = get_transforms_from_cfg(cfg, processor)

    if cfg.get("batch_augmentation", False) and "MixUp" in cfg.batch_augmentation and "CutMix" in cfg.batch_augmentation:
        m_alpha = cfg.batch_augmentation.MixUp.get("alpha", 1.0)
        c_alpha = cfg.batch_augmentation.CutMix.get("alpha", 1.0)
        mixup = v2.MixUp(num_classes=cfg.model.num_classes, alpha=m_alpha)
        cutmix = v2.CutMix(num_classes=cfg.model.num_classes, alpha=c_alpha)

        batch_aug = v2.RandomChoice([mixup, cutmix])
    else:
        batch_aug = None
    return transform, eval_transform, batch_aug

if __name__ == "__main__":
    test_cfg = OmegaConf.load("config.yaml")
    a = flatten_omegaconf(test_cfg, resolve=True)
    print(a)

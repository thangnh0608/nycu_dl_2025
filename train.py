import os
import argparse
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from omegaconf import OmegaConf
import numpy as np
import random
from datetime import datetime
import mlflow
import math
from timm.optim import create_optimizer_v2
from ema import ModelEmaV3
from utils import make_optimizer_with_layerwise_lr, save_experiment_artifacts, set_seed, flatten_omegaconf


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]


def setup_logger(log_path="train.log"):
    logger = logging.getLogger("trainer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_hf_model(model_name, num_classes, cfg):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model_params = cfg.model.get("model_params", {}).get(model_name, {})

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        **model_params
    )
    return processor, model


def train_one_epoch(model, ema, dataloader, processor, optimizer, scheduler, criterion, device, scaler, use_amp=True, use_tqdm=True):
    model.train()
    total, correct = 0, 0
    running_loss = 0.0

    iterator = tqdm(dataloader, desc="Train", leave=False) if use_tqdm else dataloader

    for imgs, labels in iterator:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            # batch = processor(images=imgs, return_tensors="pt").to(device)
            outputs = model(imgs)
            loss = criterion(outputs.logits, labels)

        # Scales loss and calls backward()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clipping)
        scaler.step(optimizer)
        scaler.update()

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:


        scheduler.step()
        ema.update(model)

        preds = outputs.logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        running_loss += loss.item() * len(labels)
        if use_tqdm:
            iterator.set_postfix({
                "train loss": f"{running_loss / total:.4f}",
                "train acc": f"{correct / total:.4f}"
            })

    acc = correct / total
    epoch_loss = running_loss / total
    logger.info(f"Train Acc: {acc:.4f}")

    return acc, epoch_loss


def evaluate(model, dataloader, processor, device, name="Val", use_tqdm=True):
    model.eval()
    total, correct = 0, 0

    iterator = tqdm(dataloader, desc=name, leave=False) if use_tqdm else dataloader

    with torch.no_grad():
        for imgs, labels in iterator:
            imgs, labels = imgs.to(device), labels.to(device)
            # batch = processor(images=imgs, return_tensors="pt", ).to(device)
            outputs = model(imgs)

            preds = outputs.logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if use_tqdm:
                iterator.set_postfix({"batch_acc": f"{correct / total:.4f}"})

    acc = correct / total
    logger.info(f"{name} Acc: {acc:.4f}")

    mlflow.log_metric(f"{name.lower()}_acc", acc)

    return acc

def make_linear_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=0.0):
    """
    Linear warmup + linear decay scheduler with minimum LR.

    Args:
        optimizer: torch optimizer
        warmup_steps: int, steps for linear warmup
        total_steps: int, total training steps
        min_lr: float, minimum learning rate factor (0.0 = zero LR)
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup from 0 to 1
            return min_lr + (1 - min_lr) * float(current_step) / float(max(1, warmup_steps))
        else:
            # linear decay from 1 to min_lr
            decay_steps = total_steps - warmup_steps
            if decay_steps <= 0:
                return min_lr
            return min_lr + (1 - min_lr) * max(0.0, float(total_steps - current_step) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def make_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, cycles=1, min_lr=0.0):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay with multiple cycles.

    Args:
        optimizer: torch.optim.Optimizer
        warmup_steps: int, number of steps to warm up
        total_steps: int, total number of training steps
        cycles: int or float, number of cosine cycles
        min_lr: float, minimum learning rate at the bottom of the cosine

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay with cycles
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + math.cos(math.pi * 2 * cycles * progress))

    return LambdaLR(optimizer, lr_lambda)

def run_inference(device, processor, model, ema, class_to_idx, test_dl):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, filenames in test_dl:
            imgs = imgs.to(device)
            # batch = processor(images=imgs, return_tensors="pt", ).to(device)
            outputs = ema.module(imgs)
            preds = outputs.logits.argmax(1).cpu().tolist()

            for f, p in zip(filenames, preds):
                results.append((f[:-4], class_to_idx[p]))
    return results


def get_lr_scheduler(cfg, optimizer, total_steps):
    scheduler_type = cfg.training.lr_scheduler
    params = cfg.training.lr_scheduler_params[scheduler_type]

    if scheduler_type == "cosine_warmup":
        scheduler = make_cosine_warmup_scheduler(
                    optimizer,
                    warmup_steps=params.warmup_steps,
                    total_steps=total_steps,
                    cycles=params.cycles,
                    min_lr=params.min_lr,
                )
    elif scheduler_type == "linear":
        scheduler = make_linear_warmup_scheduler(
                    optimizer,
                    warmup_steps=params.warmup_steps,
                    total_steps=total_steps,
                    min_lr=params.min_lr,
                )
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_type}")

    return scheduler

if __name__ == "__main__":
    args = get_args()
    cfg = OmegaConf.load(args.config)
    if os.path.isdir(os.path.join(cfg.experiment.save_dir, cfg.experiment.run_name)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.experiment.run_name = f"{cfg.experiment.run_name}_{timestamp}"
    set_seed(cfg.training.seed)
    save_path = os.path.join(cfg.experiment.save_dir, cfg.experiment.run_name)
    os.makedirs(save_path, exist_ok=True)
    logger = setup_logger(os.path.join(save_path, "train.log"))

    OmegaConf.save(cfg, os.path.join(save_path, "config.yaml"))
    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(cfg))
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = cfg.training.get("use_amp", False)
        scaler = torch.amp.GradScaler(enabled=use_amp)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(cfg.experiment.project)

        with mlflow.start_run(run_name=cfg.experiment.run_name):
            mlflow.log_artifact(__file__, artifact_path="source_code")
            mlflow.autolog()

            # log config
            mlflow.log_params(flatten_omegaconf(cfg, resolve=True))

            # Model
            processor, model = load_hf_model(cfg.model.name, cfg.model.num_classes, cfg)
            if cfg.model.get('weight', None) is not None:
                weight = torch.load(cfg.model.weight, map_location=device)
                model.load_state_dict(weight['model_state_dict'])
                logger.info(f"Loaded checkpoint {cfg.model.weight}")

            model = model.to(device)
            ema_decay = cfg.training.get("ema_decay", 0.999)
            ema = ModelEmaV3(
                model,
                decay=ema_decay,
                device=device,
                exclude_buffers=False
            )

            ema.to(device)
            # optimizer = make_optimizer_with_layerwise_lr(model, cfg)
            optimizer = create_optimizer_v2(
                model,
                opt="adamw",
                lr=cfg.training.lr,
                weight_decay=cfg.training.w_decay,
                layer_decay=cfg.training.layerwise_lr_decay
            )

            criterion = nn.CrossEntropyLoss()

            # transforms
            transform = transforms.Compose([
                transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
                # transforms.RandomResizedCrop(
                #     size=cfg.dataset.image_size,
                #     scale=(0.9, 1.0),
                #     ratio=(0.95, 1.05)
                # ),
                # transforms.RandomRotation(degrees=10),
                # transforms.ColorJitter(
                #     brightness=0.15,
                #     contrast=0.15,
                #     saturation=0.15,
                #     hue=0.05
                # ),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
            ])

            eval_transform = transforms.Compose([
                transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
            ])

            # # Dataset
            train_ds = datasets.ImageFolder(cfg.dataset.train_dir, transform=transform)
            train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
            do_val = cfg.training.get('val', False)
            if do_val:
                val_ds = datasets.ImageFolder(cfg.dataset.val_dir, transform=eval_transform)
                val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                                    shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

            test_ds = TestDataset(cfg.dataset.test_dir, transform=eval_transform)
            test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                                shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

            steps_per_epoch = len(train_dl)
            total_steps = steps_per_epoch * cfg.training.epochs
            warmup_steps = int(cfg.training.warmup * total_steps)
            scheduler = get_lr_scheduler(cfg, optimizer, total_steps)

            best_val_acc = 0.0
            best_model_path = os.path.join(save_path, "best_val_model_ema.pth")
            for epoch in range(cfg.training.epochs):
                logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")
                train_acc, train_loss = train_one_epoch(model, ema, train_dl, processor, optimizer, scheduler, criterion, device, scaler)
                if do_val:
                    val_acc = evaluate(ema.module, val_dl, processor, device, name="Val-EMA", use_tqdm=True)
                    if  val_acc > best_val_acc:
                        best_val_acc = val_acc
                        logger.info(f"New best Val-EMA Acc: {best_val_acc:.4f} — Saving best model")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': ema.module.state_dict(),
                            'val_acc': val_acc,
                            'class_to_idx': train_ds.class_to_idx,
                        }, best_model_path)

                        mlflow.log_artifact(best_model_path, artifact_path="models")
                    val_acc = evaluate(ema.module, val_dl, processor, device, name="Val-EMA", use_tqdm=True)
                    mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("lr", optimizer.param_groups[-1]['lr'], step=epoch)

            # Testing
            print(os.path.join(save_path, "last_model_ema.pth"))
            print(save_path)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ema.module.state_dict(),
                'class_to_idx': train_ds.class_to_idx,
            }, os.path.join(save_path, "last_model_ema.pth"))

            logger.info("Running inference for test data")
            results_last = run_inference(device, processor, ema.module, ema, train_ds.classes, test_dl)
            if os.path.isfile(best_model_path):
                best_checkpoint = torch.load(best_model_path, map_location=device)
                ema.module.load_state_dict(best_checkpoint['model_state_dict'])
                results_best = run_inference(device, processor, ema.module, ema, train_ds.classes, test_dl)
                save_experiment_artifacts(results_best, save_path, logger=None, suffix=f'{os.path.basename(cfg.experiment.run_name)}_best')

            save_experiment_artifacts(results_last, logger, save_path, suffix=f'{os.path.basename(cfg.experiment.run_name)}_last')
            logger.info("Experiment finished")
    except Exception as e:
        logger.error("Error")
        logger.error(e, exc_info=True)
        raise e

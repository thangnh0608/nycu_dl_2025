import os
import argparse
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, datasets
from PIL import Image
from transformers.modeling_outputs import ImageClassifierOutput
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from omegaconf import OmegaConf
from torchvision.transforms import v2
import numpy as np
import random
from datetime import datetime
import mlflow
import math
from timm.optim import create_optimizer_v2
from ema import ModelEmaV3
from utils import (
    make_optimizer_with_layerwise_lr,
    save_experiment_artifacts,
    set_seed,
    flatten_omegaconf,
    get_transforms_from_cfg,
    run_inference,
    apply_overrides
)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FFTMag(v2.Transform):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=0, keepdim=True)

        fft = torch.fft.fft2(x, norm="ortho")
        mag = torch.abs(fft)
        mag = torch.log1p(mag)
        mag = torch.fft.fftshift(mag, dim=(-2, -1))

        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)

        mag = (mag - mean) / std
        return mag

class RGBFFTDataset(datasets.ImageFolder):
    def __init__(self, root, rgb_transform):
        super().__init__(root)
        self.rgb_transform = rgb_transform
        self.freq_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            FFTMag(),
        ])

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)

        return {
            "rgb": self.rgb_transform(img),
            "fft": self.freq_transform(img),
            "label": label,
        }

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = sorted(os.listdir(root_dir))
        self.transform = transform
        self.freq_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            FFTMag(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        fft = self.freq_transform(image)
        return image, fft, self.files[idx]

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNextV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = inputs + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNextV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class HFConvNeXt_FFT_ConvNeXtFusion(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.rgb_model = hf_model
        fft_branch_depths = [2, 2, 8, 2]
        fft_branch_dims = [80, 160, 320, 640]
        self.fft_branch =  ConvNeXtV2(depths=fft_branch_depths, dims=fft_branch_dims, drop_path_rate=0.2)
        rgb_dim = hf_model.config.num_features
        self.classifier = nn.Linear(rgb_dim + fft_branch_dims[-1], hf_model.config.num_labels)

    def forward(self, pixel_values_rgb, fft_values):
        outputs = self.rgb_model(
            pixel_values=pixel_values_rgb,
            output_hidden_states=True,
            return_dict=True
        )
        f_rgb = outputs.hidden_states[-1]
        if f_rgb.ndim == 4:
            f_rgb = f_rgb.mean(dim=(-2,-1))

        f_fft = self.fft_branch(fft_values)

        # Fusion
        fused = torch.cat([f_rgb, f_fft], dim=1)
        logits = self.classifier(fused)

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=fused
        )

def load_hf_model_with_fft(model_name, num_classes, cfg):
    model_cfg = AutoConfig.from_pretrained(cfg.model.name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    model_params = cfg.model.get("model_params", {}).get(model_name, {})
    if model_params:
        model_params = OmegaConf.to_container(model_params, resolve=True)
        apply_overrides(model_cfg, model_params)

    model_cfg.num_labels = num_classes

    # Load real HF model
    hf_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        config=model_cfg,
        ignore_mismatched_sizes=True
    )
    # Freeze logic
    freeze_hf_model_by_cfg(cfg, hf_model)

    # Wrap with FFT branch fusion
    model = HFConvNeXt_FFT_ConvNeXtFusion(hf_model)

    return processor, model

def freeze_hf_model_by_cfg(cfg, hf_model):
    N = cfg.model.get("freeze", 1)
    for i, block in enumerate(hf_model.timm_model.stages):
        if i < N:
            for p in block.parameters():
                p.requires_grad = False
        else:
            for p in block.parameters():
                p.requires_grad = True

    for p in hf_model.timm_model.stem.parameters():
        p.requires_grad = False
    for p in hf_model.timm_model.head.parameters():
        p.requires_grad = True

def run_inference_fft(device, model, ema, class_to_idx, test_dl):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, fft, filenames in test_dl:
            imgs = imgs.to(device)
            fft = fft.to(device)
            # batch = processor(images=imgs, return_tensors="pt", ).to(device)
            outputs = ema.module(imgs, fft)
            score = torch.softmax(outputs.logits, dim=1).cpu().tolist()
            preds = outputs.logits.argmax(1).cpu().tolist()

            for f, p, s in zip(filenames, preds, score):
                results.append((f[:-4], class_to_idx[p], s))
    return results



if __name__ == "__main__":
    B, C, H, W = 2, 3, 224, 224
    dummy_rgb = torch.rand(B, C, H, W)

    # Example: ConvNeXt DINOv3 checkpoint name
    model_name = "timm/convnext_large.dinov3_lvd1689m"
    num_classes = 2
    cfg = type("", (), {})()  # dummy cfg object
    cfg.model = type("", (), {})()
    cfg.model.name = model_name
    cfg.model.get = lambda key, default=None: default

    processor, model = load_hf_model_with_fft(model_name, num_classes, cfg)
    logits = model(dummy_rgb)
    print("Logits shape:", logits.shape)
    print(logits)

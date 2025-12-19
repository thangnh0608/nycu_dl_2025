import os
import subprocess
import itertools
from omegaconf import OmegaConf
from datetime import datetime

# ==================== CONFIGURATION ====================
BASE_CONFIG_PATH = "config.yaml"           # your base config file
TRAIN_SCRIPT = "train.py"                  # your training script
CONFIGS_DIR = "configs/temp"
os.makedirs(CONFIGS_DIR, exist_ok=True)

# Models to sweep — feel free to add/remove
PARAM_GRID = {
    "model.name": [
        "facebook/convnextv2-base-22k-224",
        "google/vit-base-patch16-224",
        "google/vit-large-patch16-224",
        "microsoft/swin-tiny-patch4-window7-224",
        "microsoft/swin-base-patch4-window7-224",
        "timm/efficientnet_b0.ra_in1k",
        "timm/efficientnet_b3.ra2_in1k",
        "facebook/deit-base-patch16-224",
    ],
}

# Load base config
if not os.path.exists(BASE_CONFIG_PATH):
    raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}")

base_cfg = OmegaConf.load(BASE_CONFIG_PATH)
print(f"Loaded base config from: {BASE_CONFIG_PATH}")
print("Base run_name:", base_cfg.experiment.run_name)
print("=" * 60)

# Generate all combinations (just models for now)
keys, values = zip(*PARAM_GRID.items())
combinations = list(itertools.product(*values))
total_runs = len(combinations)
print(f"Total experiments: {total_runs}\n")

for i, combo in enumerate(combinations):
    overrides = dict(zip(keys, combo))
    run_cfg = OmegaConf.merge(base_cfg, overrides)

    # Extract info for unique naming
    model_name_short = run_cfg.model.name.split("/")[-1]  # e.g., convnextv2-base-22k-224
    lr = run_cfg.training.lr
    bs = run_cfg.training.batch_size
    ep = run_cfg.training.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"{model_name_short}_lr{lr}_bs{bs}_ep{ep}_{timestamp}"

    # Unique output directory per run
    run_cfg.experiment.run_name = f"{base_cfg.experiment.run_name}/{suffix}"

    # Unique config file
    config_filename = f"config_run_{i:03d}_{suffix}.yaml"
    config_path = os.path.join(CONFIGS_DIR, config_filename)
    OmegaConf.save(run_cfg, config_path)

    print(f"Run {i+1:02d}/{total_runs}")
    print(f"   Model       : {run_cfg.model.name}")
    print(f"   Hyperparams : lr={lr}, batch_size={bs}, epochs={ep}")
    print(f"   Output dir  : {run_cfg.experiment.run_name}")
    print(f"   Config file : {config_path}")

    # Launch with uv (clean, no Kaggle junk)
    cmd = ["uv", "run", TRAIN_SCRIPT, "--config", config_path]
    print(f"   Command     : {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"   → SUCCESS!\n")
    else:
        print(f"   → FAILED (code: {result.returncode})\n")

print("   mlflow ui --backend-store-uri file:./mlruns")
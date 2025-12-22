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

# Hyperparameter grid to sweep — model fixed to google/vit-base-patch16-224
# Batch size is NOT tuned (kept fixed as in base config)
PARAM_GRID = {
    "training.w_decay": [0.05, 0.1],
    "layerwise_lr_decay": [0.3, 0.3, 0.7],
}

# Load base config
if not os.path.exists(BASE_CONFIG_PATH):
    raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}")

base_cfg = OmegaConf.load(BASE_CONFIG_PATH)
print(f"Loaded base config from: {BASE_CONFIG_PATH}")
print("Base run_name:", base_cfg.experiment.run_name)
print("=" * 60)

# Generate all combinations
keys, values = zip(*PARAM_GRID.items())
combinations = list(itertools.product(*values))
total_runs = len(combinations)
print(f"Total experiments: {total_runs}\n")

for i, combo in enumerate(combinations):
    overrides = dict(zip(keys, combo))
    run_cfg = OmegaConf.merge(base_cfg, overrides)

    # Extract info for unique naming (model is fixed)
    model_name_short = run_cfg.model.name.split("/")[-1]  # e.g., vit-base-patch16-224
    lr = run_cfg.training.lr
    ep = run_cfg.training.epochs
    wd = run_cfg.training.w_decay
    # h_drop = run_cfg.model.model_params["google/vit-base-patch16-224"].hidden_dropout_prob
    # a_drop = run_cfg.model.model_params["google/vit-base-patch16-224"].attention_probs_dropout_prob
    bs = run_cfg.training.batch_size  # included for info, but not tuned

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"{model_name_short}_lr{lr}_ep{ep}_wd{wd}_{timestamp}"

    # Unique output directory per run
    run_cfg.experiment.run_name = f"{base_cfg.experiment.run_name}/{suffix}"

    # Unique config file
    config_filename = f"config_run_{i:03d}_{suffix}.yaml"
    config_path = os.path.join(CONFIGS_DIR, config_filename)
    OmegaConf.save(run_cfg, config_path)

    print(f"Run {i+1:03d}/{total_runs}")
    print(f"   Model       : {run_cfg.model.name}")
    print(f"   Hyperparams : lr={lr}, epochs={ep}, w_decay={wd}")
    print(f"                 batch_size={bs} (fixed)")
    print(f"   Output dir  : {run_cfg.experiment.run_name}")
    print(f"   Config file : {config_path}")

    # Launch with uv
    cmd = ["python", TRAIN_SCRIPT, "--config", config_path]
    print(f"   Command     : {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"   → SUCCESS!\n")
    else:
        print(f"   → FAILED (code: {result.returncode})\n")

print("All runs completed!")
print("To view results: mlflow ui --backend-store-uri file:./mlruns")
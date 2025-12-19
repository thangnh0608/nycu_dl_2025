import os
import argparse
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ema import ModelEmaV3
# Custom Dataset for unlabeled test images
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Filter for common image extensions
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="test_predictions_05.csv")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model and Weights
    processor, model = load_hf_model(cfg.model.name, cfg.model.num_classes, cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    ema = ModelEmaV3(
                model,
                decay=cfg.training.get("ema_decay", 0.999),
                device=device,
                exclude_buffers=False
            )
    ema.module.load_state_dict(checkpoint['model_state_dict'])

    # 2. Setup Class Mapping
    # Logic: Find which index is 'real', the other is 'fake'
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    real_idx = None
    fake_idx = None

    for idx, name in idx_to_class.items():
        if name.lower() == 'real':
            real_idx = idx
        elif name.lower() == 'fake':
            fake_idx = idx

    if real_idx is None or fake_idx is None:
        raise ValueError("Could not detect 'real' and 'fake' classes from checkpoint.")

    # 3. Data Loading
    transform_test = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.ToTensor(),
    ])

    test_ds = TestDataset(cfg.dataset.test_dir, transform=transform_test)
    test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)

    # 4. Inference with Threshold
    results = []
    print(f"Running inference on {len(test_ds)} images with threshold {args.threshold}...")

    model.eval()
    results = []
    with torch.no_grad():
        for imgs, filenames in test_dl:
            imgs = imgs.to(device)
            batch = processor(images=imgs, return_tensors="pt", do_rescale=False).to(device)
            outputs = ema.module(**batch)
            preds = outputs.logits.argmax(1).cpu().tolist()

            for i in range(len(filenames)):
                # Remove file extension for the filename column if needed, or keep it
                fname = os.path.splitext(filenames[i])[0]
                predicted_label = idx_to_class[preds[i]]

                results.append({
                    'filename': fname,
                    'label': predicted_label
                })

    # 5. Save to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
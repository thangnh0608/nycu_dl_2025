import os
import shutil
import random

# -------------------------- CONFIG --------------------------
ROOT_DIR = "data/train"                  # Original training data: data/train/fake and data/train/real
TRAIN_OUT_DIR = "data/train_split"       # New training split (80%)
VAL_OUT_DIR = "data/val"                 # Validation split (20%)

VAL_RATIO = 0.2                          # 20% for validation
RANDOM_SEED = 42                         # For reproducibility
OVERWRITE_EXISTING = False               # Set to True if you want to delete and recreate output dirs

# -----------------------------------------------------------

def safe_makedirs(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not OVERWRITE_EXISTING:
        print(f"Warning: Directory already exists: {dir_path}")
        print("         Set OVERWRITE_EXISTING = True to delete and recreate.")
    else:
        print(f"Overwriting existing directory: {dir_path}")
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def get_image_files(folder):
    """Get list of common image files in folder"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if os.path.splitext(fname.lower())[1] in valid_extensions:
                files.append(os.path.join(folder, fname))
    return files

def copy_file_safe(src, dst_dir, filename):
    """Copy file safely, avoiding name conflicts by adding _1, _2, etc."""
    base_name, ext = os.path.splitext(filename)
    dest_path = os.path.join(dst_dir, filename)
    counter = 1
    while os.path.exists(dest_path):
        new_filename = f"{base_name}_{counter}{ext}"
        dest_path = os.path.join(dst_dir, new_filename)
        counter += 1
    shutil.copy2(src, dest_path)
    return dest_path

def main():
    random.seed(RANDOM_SEED)

    # Check source directory
    if not os.path.exists(ROOT_DIR):
        raise FileNotFoundError(f"Source directory not found: {ROOT_DIR}")

    classes = ["fake", "real"]
    class_paths = {cls: os.path.join(ROOT_DIR, cls) for cls in classes}

    for cls, path in class_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Class folder not found: {path}")

    # Prepare output directories
    print("Preparing output directories...")
    safe_makedirs(TRAIN_OUT_DIR)
    safe_makedirs(VAL_OUT_DIR)
    for cls in classes:
        safe_makedirs(os.path.join(TRAIN_OUT_DIR, cls))
        safe_makedirs(os.path.join(VAL_OUT_DIR, cls))

    print("\nSplitting dataset (stratified, 80% train / 20% val)...\n")

    total_train = 0
    total_val = 0

    for cls in classes:
        class_dir = class_paths[cls]
        images = get_image_files(class_dir)

        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue

        print(f"Class '{cls}': {len(images)} images")
        random.shuffle(images)

        val_count = int(len(images) * VAL_RATIO)
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Copy training images
        train_dest_dir = os.path.join(TRAIN_OUT_DIR, cls)
        for img_path in train_images:
            filename = os.path.basename(img_path)
            copy_file_safe(img_path, train_dest_dir, filename)
        total_train += len(train_images)

        # Copy validation images
        val_dest_dir = os.path.join(VAL_OUT_DIR, cls)
        for img_path in val_images:
            filename = os.path.basename(img_path)
            copy_file_safe(img_path, val_dest_dir, filename)
        total_val += len(val_images)

        print(f"  → Train: {len(train_images)}, Val: {len(val_images)}")

    print("\n" + "="*50)
    print("Dataset split completed successfully!")
    print(f"Training images  : {total_train} → {TRAIN_OUT_DIR}")
    print(f"Validation images: {total_val} → {VAL_OUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
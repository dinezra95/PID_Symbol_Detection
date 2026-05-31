"""
Prepare pre-split train/test data for YOLO training.

Steps:
  1. Generate patches (1024x1024 with overlap) from train and test images
  2. Organize into YOLO directory structure (train / val / test)
  3. Create dataset.yaml

Usage:
  cd src
  python prepare_training_data.py
"""

import sys
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from utils.patch_generator import PatchGenerator
from utils.helpers import get_files

# ── Config ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

RAW_TRAIN_IMAGES = ROOT / "data/raw/sheets/train"
RAW_TRAIN_LABELS = ROOT / "data/raw/labels/train"
RAW_TEST_IMAGES  = ROOT / "data/raw/sheets/test"
RAW_TEST_LABELS  = ROOT / "data/raw/labels/test"

PATCHES_TRAIN = ROOT / "data/processed/stage_1/patches/train"
PATCHES_TEST  = ROOT / "data/processed/stage_1/patches/test"

YOLO_OUTPUT = ROOT / "data/processed/stage_1/yolo_class_agnostic"

PATCH_SIZE = (1024, 1024)
PATCH_OVERLAP = (50, 50)
MIN_VISIBILITY = 0.3
VAL_RATIO = 0.8  # 80% of test images → val, 20% → test
RANDOM_SEED = 42
CLASS_NAMES = ["Symbol"]


def step1_generate_patches():
    print("=" * 60)
    print("Step 1: Generating patches")
    print("=" * 60)

    pg = PatchGenerator(
        patch_size=PATCH_SIZE,
        overlap=PATCH_OVERLAP,
        min_visibility=MIN_VISIBILITY,
    )

    print(f"\n  Train: {RAW_TRAIN_IMAGES} → {PATCHES_TRAIN}")
    pg.generate_patches(
        images_dir=RAW_TRAIN_IMAGES,
        labels_dir=RAW_TRAIN_LABELS,
        output_dir=PATCHES_TRAIN,
    )

    print(f"\n  Test:  {RAW_TEST_IMAGES} → {PATCHES_TEST}")
    pg.generate_patches(
        images_dir=RAW_TEST_IMAGES,
        labels_dir=RAW_TEST_LABELS,
        output_dir=PATCHES_TEST,
    )
    print()


def step2_build_yolo_structure():
    print("=" * 60)
    print("Step 2: Building YOLO directory structure")
    print("=" * 60)

    for split in ("train", "val", "test"):
        (YOLO_OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Train patches → YOLO train ──
    train_images = get_files(PATCHES_TRAIN, [".jpg", ".png", ".jpeg"])
    print(f"\n  Copying {len(train_images)} train patches...")
    for img in train_images:
        lbl = img.with_suffix(".txt")
        shutil.copy2(img, YOLO_OUTPUT / "images/train" / img.name)
        if lbl.exists():
            shutil.copy2(lbl, YOLO_OUTPUT / "labels/train" / lbl.name)

    # ── Test patches → YOLO val + test (grouped by source image) ──
    test_images = get_files(PATCHES_TEST, [".jpg", ".png", ".jpeg"])
    groups = defaultdict(list)
    for img in test_images:
        # patch name format: {source_image}_{row}_{col}.jpg
        # group key = source image id (everything before last two _N_N)
        parts = img.stem.rsplit("_", 2)
        source_key = parts[0] if len(parts) == 3 else img.stem
        groups[source_key].append(img)

    group_keys = sorted(groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(group_keys)

    split_idx = int(len(group_keys) * VAL_RATIO)
    val_keys = group_keys[:split_idx]
    test_keys = group_keys[split_idx:]

    val_patches = [p for k in val_keys for p in groups[k]]
    test_patches = [p for k in test_keys for p in groups[k]]

    print(f"  Test images split: {len(val_keys)} source images → val ({len(val_patches)} patches), "
          f"{len(test_keys)} source images → test ({len(test_patches)} patches)")

    for img in val_patches:
        lbl = img.with_suffix(".txt")
        shutil.copy2(img, YOLO_OUTPUT / "images/val" / img.name)
        if lbl.exists():
            shutil.copy2(lbl, YOLO_OUTPUT / "labels/val" / lbl.name)

    for img in test_patches:
        lbl = img.with_suffix(".txt")
        shutil.copy2(img, YOLO_OUTPUT / "images/test" / img.name)
        if lbl.exists():
            shutil.copy2(lbl, YOLO_OUTPUT / "labels/test" / lbl.name)

    print()


def step3_create_dataset_yaml():
    print("=" * 60)
    print("Step 3: Creating dataset.yaml")
    print("=" * 60)

    data = {
        "train": str((YOLO_OUTPUT / "images/train").resolve()),
        "val": str((YOLO_OUTPUT / "images/val").resolve()),
        "test": str((YOLO_OUTPUT / "images/test").resolve()),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = YOLO_OUTPUT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\n  Saved: {yaml_path}")
    print(f"  Contents: {data}")
    print()


def print_summary():
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for split in ("train", "val", "test"):
        imgs = list((YOLO_OUTPUT / "images" / split).glob("*"))
        lbls = list((YOLO_OUTPUT / "labels" / split).glob("*"))
        print(f"  {split:>5}: {len(imgs)} images, {len(lbls)} labels")
    print(f"\n  dataset.yaml: {YOLO_OUTPUT / 'dataset.yaml'}")
    print("\nDone! You can now train with:")
    print("  python src/run_pipeline.py stage1 --train_model")


if __name__ == "__main__":
    step1_generate_patches()
    step2_build_yolo_structure()
    step3_create_dataset_yaml()
    print_summary()

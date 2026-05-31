"""
Crop detected symbols from Stage 1 results.

Usage:
  python src/crop_detections.py \
    --images /path/to/original/images \
    --labels /path/to/stage1_results \
    --output /path/to/crops_output
"""

import argparse
import cv2
from pathlib import Path


def crop_detections(images_dir, labels_dir, output_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    total_crops = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name} (cannot read)")
            continue

        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  No labels for {img_path.name}")
            continue

        img_output_dir = output_dir / img_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = img_output_dir / f"crop_{i:03d}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_crops += 1

        print(f"  {img_path.stem}: {len(lines)} crops")

    print(f"\nDone! {total_crops} crops saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop detected symbols from Stage 1 results")
    parser.add_argument("--images", type=str, required=True, help="Directory with original images")
    parser.add_argument("--labels", type=str, required=True, help="Directory with Stage 1 YOLO labels")
    parser.add_argument("--output", type=str, required=True, help="Output directory for crops")
    args = parser.parse_args()

    crop_detections(args.images, args.labels, args.output)

"""
Compare predictions from two models side by side on the same image.
Draws bounding boxes from each model in a different color.

Usage:
  python src/compare_predictions.py \
    --images /path/to/images \
    --pred_a /path/to/results_a \
    --pred_b /path/to/results_b \
    --label_a "YOLOv8n" \
    --label_b "YOLO11n" \
    --output /path/to/comparison
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


COLOR_A = (255, 70, 70)    # red (BGR)
COLOR_B = (70, 200, 70)    # green (BGR)


def load_yolo_boxes(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes, color, thickness=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_legend(img, label_a, label_b):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(h, w) / 2000)
    thick = max(1, int(scale * 2))
    y_start = int(30 * scale) + 10

    cv2.rectangle(img, (10, 5), (int(250 * scale) + 20, y_start * 2 + 15), (255, 255, 255), -1)
    cv2.rectangle(img, (10, 5), (int(250 * scale) + 20, y_start * 2 + 15), (0, 0, 0), 1)

    cv2.putText(img, label_a, (20, y_start), font, scale, COLOR_A, thick)
    cv2.putText(img, label_b, (20, y_start * 2), font, scale, COLOR_B, thick)


def compare(images_dir, pred_a_dir, pred_b_dir, label_a, label_b, output_dir):
    images_dir = Path(images_dir)
    pred_a_dir = Path(pred_a_dir)
    pred_b_dir = Path(pred_b_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Comparing {len(image_files)} images...\n")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name} (cannot read)")
            continue

        h, w = img.shape[:2]
        stem = img_path.stem

        boxes_a = load_yolo_boxes(pred_a_dir / f"{stem}.txt", w, h)
        boxes_b = load_yolo_boxes(pred_b_dir / f"{stem}.txt", w, h)

        draw_boxes(img, boxes_a, COLOR_A, thickness=2)
        draw_boxes(img, boxes_b, COLOR_B, thickness=2)
        draw_legend(img, f"{label_a} ({len(boxes_a)})", f"{label_b} ({len(boxes_b)})")

        out_path = output_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {stem}: {label_a}={len(boxes_a)} boxes, {label_b}={len(boxes_b)} boxes")

    print(f"\nDone! Comparisons saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predictions from two models")
    parser.add_argument("--images", type=str, required=True, help="Directory with original images")
    parser.add_argument("--pred_a", type=str, required=True, help="Predictions directory for model A")
    parser.add_argument("--pred_b", type=str, required=True, help="Predictions directory for model B")
    parser.add_argument("--label_a", type=str, default="Model A", help="Label for model A")
    parser.add_argument("--label_b", type=str, default="Model B", help="Label for model B")
    parser.add_argument("--output", type=str, required=True, help="Output directory for comparison images")
    args = parser.parse_args()

    compare(args.images, args.pred_a, args.pred_b, args.label_a, args.label_b, args.output)

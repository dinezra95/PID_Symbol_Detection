"""
Visualize Stage 2 results: draw bounding boxes with class labels on images.

Usage:
  python src/visualize_stage2.py \
    --images /path/to/images \
    --labels /path/to/stage2_results \
    --classes /path/to/classes.txt \
    --output /path/to/output
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (60, 100, 180), (200, 80, 120), (80, 200, 120), (120, 80, 200),
    (200, 200, 80), (80, 200, 200), (200, 80, 200), (100, 150, 50),
    (50, 100, 150), (150, 50, 100), (180, 180, 60), (60, 180, 180),
    (180, 60, 180), (100, 200, 100),
]


def load_classes(classes_path):
    classes = {}
    with open(classes_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            cid, name = line.split(":", 1)
            classes[int(cid.strip())] = name.strip()
    return classes


def load_labels(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def get_color(cls_id):
    return COLORS[cls_id % len(COLORS)]


def visualize(images_dir, labels_dir, classes_path, output_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_path)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Visualizing {len(image_files)} images...\n")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name}")
            continue

        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = load_labels(label_path, w, h)

        scale = max(0.4, min(h, w) / 3000)
        thickness = max(1, int(scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, scale * 0.8)

        class_counts = defaultdict(int)
        for (cls, x1, y1, x2, y2) in boxes:
            color = get_color(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = f"{cls}"
            label_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 3), font, font_scale, (255, 255, 255), 1)

            class_counts[cls] += 1

        # Draw legend
        legend_items = sorted(class_counts.items())
        if legend_items:
            line_h = int(20 * scale) + 10
            legend_h = len(legend_items) * line_h + 20
            legend_w = int(300 * scale) + 20
            cv2.rectangle(img, (5, 5), (legend_w, legend_h), (255, 255, 255), -1)
            cv2.rectangle(img, (5, 5), (legend_w, legend_h), (0, 0, 0), 1)

            for i, (cls, count) in enumerate(legend_items):
                y = 15 + (i + 1) * line_h - 5
                color = get_color(cls)
                name = classes.get(cls, f"Class {cls}")
                cv2.rectangle(img, (10, y - int(10 * scale)), (10 + int(15 * scale), y + 2), color, -1)
                cv2.putText(img, f"{cls}: {name} ({count})", (15 + int(15 * scale), y),
                            font, font_scale * 0.9, (0, 0, 0), 1)

        out_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {img_path.stem}: {len(boxes)} detections, {len(class_counts)} classes")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Stage 2 results")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    visualize(args.images, args.labels, args.classes, args.output)

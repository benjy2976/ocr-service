#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Export stamp crops from YOLO labels.")
    p.add_argument("--images", default="/data/out/stamp_pages/images", help="Images dir")
    p.add_argument("--labels", default="/data/out/stamp_pages/labels", help="Labels dir")
    p.add_argument("--out", default="/data/classify/crops", help="Output crops dir")
    p.add_argument("--index", default="/data/classify/index.csv", help="Index CSV path")
    p.add_argument("--min-size", type=int, default=10, help="Minimum crop size in pixels")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.exists():
            image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            bw = int(w * w_img)
            bh = int(h * h_img)
            x = int(cx * w_img - bw / 2)
            y = int(cy * h_img - bh / 2)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w_img - x)
            bh = min(bh, h_img - y)
            if bw < args.min_size or bh < args.min_size:
                continue
            crop = img[y : y + bh, x : x + bw]
            out_name = f"{label_path.stem}_b{idx}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop)
            rows.append(
                {
                    "crop": out_name,
                    "source_image": image_path.name,
                    "box_index": idx,
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                }
            )

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["crop", "source_image", "box_index", "x", "y", "w", "h"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Crops: {len(rows)}")
    print(f"Out: {out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build YOLO text-detection dataset from reviewed or auto labels.")
    p.add_argument("--images", default="/data/out/text_pages/images", help="Text pages images dir")
    p.add_argument(
        "--labels-reviewed",
        default="/data/out/text_pages/labels_reviewed",
        help="Reviewed labels dir",
    )
    p.add_argument(
        "--labels-auto",
        default="/data/out/text_pages/labels_auto",
        help="Auto labels dir used as fallback",
    )
    p.add_argument("--out", default="/data/datasets/text_blocks_detect", help="Dataset output dir")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument(
        "--include-empty",
        action="store_true",
        help="Include pages without labels as negatives",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images)
    reviewed_dir = Path(args.labels_reviewed)
    auto_dir = Path(args.labels_auto)
    out_dir = Path(args.out)
    out_train_images = out_dir / "images" / "train"
    out_val_images = out_dir / "images" / "val"
    out_train_labels = out_dir / "labels" / "train"
    out_val_labels = out_dir / "labels" / "val"
    for path in (out_train_images, out_val_images, out_train_labels, out_val_labels):
        path.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        raise SystemExit(f"No images found: {images_dir}")

    selected = []
    reviewed_count = 0
    auto_count = 0
    empty_count = 0
    for image_path in images:
        reviewed_path = reviewed_dir / f"{image_path.stem}.txt"
        auto_path = auto_dir / f"{image_path.stem}.txt"
        reviewed_lines = read_labels(reviewed_path)
        auto_lines = read_labels(auto_path)

        if reviewed_lines:
            label_lines = reviewed_lines
            source = "reviewed"
            reviewed_count += 1
        elif auto_lines:
            label_lines = auto_lines
            source = "auto"
            auto_count += 1
        else:
            label_lines = []
            source = "empty"
            empty_count += 1
            if not args.include_empty:
                continue

        selected.append((image_path, label_lines, source))

    if not selected:
        raise SystemExit("No labeled pages available.")

    random.seed(args.seed)
    random.shuffle(selected)
    val_count = max(1, int(len(selected) * args.val_ratio))
    val_names = {item[0].name for item in selected[:val_count]}

    used = 0
    for image_path, label_lines, _source in selected:
        is_val = image_path.name in val_names
        out_image = (out_val_images if is_val else out_train_images) / image_path.name
        out_label = (out_val_labels if is_val else out_train_labels) / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        used += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: /data/datasets/text_blocks_detect\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['text_block']\n"
    )

    report = {
        "images_total": len(images),
        "images_used": used,
        "reviewed_labels": reviewed_count,
        "auto_labels": auto_count,
        "empty_labels": empty_count,
        "include_empty": args.include_empty,
        "val_images": len(val_names),
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"Images used: {used}/{len(images)}")
    print(f"Reviewed labels: {reviewed_count}")
    print(f"Auto labels: {auto_count}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2


CLASSES = [
    "sello_redondo",
    "logo",
    "firma",
    "firma_con_huella",
    "sello_completo",
    "sello_cuadrado",
    "huella_digital",
    "sello_proveido",
    "sello_recepcion",
    "sello_fedatario",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build detection dataset from stamp pages + classified crops.")
    p.add_argument("--pages", default="/data/out/stamp_pages/images", help="Pages images dir")
    p.add_argument("--page-labels", default="/data/out/stamp_pages/labels", help="Page labels dir (YOLO)")
    p.add_argument("--index", default="/data/classify/index.csv", help="Index CSV mapping crops to pages")
    p.add_argument("--classify", default="/data/classify/state_merged.json", help="Classify state JSON")
    p.add_argument("--preds", default="/data/classify/preds.json", help="Predictions JSON")
    p.add_argument("--min-conf", type=float, default=0.95, help="Minimum confidence to include from preds")
    p.add_argument("--out", default="/data/datasets/stamps_detect", help="Output dataset dir")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pages_dir = Path(args.pages)
    labels_dir = Path(args.page_labels)
    out_dir = Path(args.out)
    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"
    out_labels_train.mkdir(parents=True, exist_ok=True)
    out_labels_val.mkdir(parents=True, exist_ok=True)
    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_val.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path}")
    crop_to_class = {}
    state_path = Path(args.classify)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        items = state.get("items", {})
        for name, meta in items.items():
            label = meta.get("label")
            status = meta.get("status")
            if not label or label in {"__skipped__", "__rejected__"}:
                continue
            if status not in {"validated"}:
                continue
            crop_to_class[name] = label

    preds_path = Path(args.preds)
    if preds_path.exists():
        preds = json.loads(preds_path.read_text())
        for name, meta in preds.items():
            if name in crop_to_class:
                continue
            label = meta.get("label")
            conf = float(meta.get("confidence", 0.0) or 0.0)
            if not label or conf < args.min_conf:
                continue
            crop_to_class[name] = label

    crop_to_page = {}
    with index_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop = row.get("crop")
            source = row.get("source_image")
            box_index = row.get("box_index")
            if crop and source and box_index is not None:
                crop_to_page[crop] = (source, int(box_index))

    pages = sorted(pages_dir.glob("*.*"))
    pages = [p for p in pages if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not pages:
        raise SystemExit("No pages found.")

    random.seed(args.seed)
    random.shuffle(pages)
    val_count = max(1, int(len(pages) * args.val_ratio))
    val_pages = set(p.name for p in pages[:val_count])

    class_to_id = {c: i for i, c in enumerate(CLASSES)}
    missing_class = 0
    missing_crop = 0
    low_conf = 0
    used_pages = 0

    for page in pages:
        label_path = labels_dir / f"{page.stem}.txt"
        if not label_path.exists():
            continue
        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        out_lines = []
        for i, line in enumerate(lines):
            # We need to map this box index to a crop and then to a class
            crop_name = f"{page.stem}_b{i}.png"
            cls_name = crop_to_class.get(crop_name)
            if not cls_name:
                missing_crop += 1
                continue
            if cls_name not in class_to_id:
                missing_class += 1
                continue
            _, cx, cy, w, h = line.split()
            out_lines.append(f"{class_to_id[cls_name]} {cx} {cy} {w} {h}")

        if not out_lines:
            continue

        if page.name in val_pages:
            out_img = out_images_val / page.name
            out_lbl = out_labels_val / f"{page.stem}.txt"
        else:
            out_img = out_images_train / page.name
            out_lbl = out_labels_train / f"{page.stem}.txt"

        if not out_img.exists():
            shutil.copy2(page, out_img)
        out_lbl.write_text("\n".join(out_lines) + "\n")
        used_pages += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        """path: /data/datasets/stamps_detect
train: images/train
val: images/val
nc: {nc}
names: {names}
""".format(nc=len(CLASSES), names=CLASSES)
    )

    report = {
        "pages_total": len(pages),
        "pages_used": used_pages,
        "val_pages": len(val_pages),
        "missing_crop_class": missing_crop,
        "missing_class_name": missing_class,
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Pages used: {used_pages}/{len(pages)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())

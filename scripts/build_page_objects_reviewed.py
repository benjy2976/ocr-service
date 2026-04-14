#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

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

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

STAMP_LABELS_DIR = Path("data/out/stamp_pages/labels")
INDEX_PATH = Path("data/classify/index.csv")
STATE_PATH = Path("data/classify/state_merged.json")
OUT_DIR = Path("data/annotations/labels/page_objects/reviewed")
META_PATH = Path("data/annotations/labels/page_objects/build_meta.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)

state = json.loads(STATE_PATH.read_text())
items = state.get("items", {})

crop_to_label = {}
for crop_name, meta in items.items():
    if meta.get("status") != "validated":
        continue
    label = meta.get("label")
    if not label or label not in CLASS_TO_ID:
        continue
    crop_to_label[crop_name] = label

page_map = defaultdict(list)

with INDEX_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        crop = row.get("crop", "").strip()
        source_image = row.get("source_image", "").strip()
        box_index_raw = row.get("box_index", "").strip()
        if crop not in crop_to_label:
            continue
        if not source_image or not box_index_raw:
            continue
        try:
            box_index = int(box_index_raw)
        except ValueError:
            continue
        page_map[source_image].append((box_index, crop_to_label[crop]))

pages_written = 0
boxes_written = 0

for image_name, entries in page_map.items():
    src_label = STAMP_LABELS_DIR / f"{Path(image_name).stem}.txt"
    if not src_label.exists():
        continue

    src_lines = [l.strip() for l in src_label.read_text().splitlines() if l.strip()]
    out_lines = []

    for box_index, label_name in sorted(entries, key=lambda x: x[0]):
        if box_index < 0 or box_index >= len(src_lines):
            continue
        parts = src_lines[box_index].split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        cls_id = CLASS_TO_ID[label_name]
        out_lines.append(f"{cls_id} {cx} {cy} {w} {h}")

    if not out_lines:
        continue

    out_path = OUT_DIR / f"{Path(image_name).stem}.txt"
    out_path.write_text("\n".join(out_lines) + "\n")
    pages_written += 1
    boxes_written += len(out_lines)

META_PATH.write_text(json.dumps({
    "pages_written": pages_written,
    "boxes_written": boxes_written,
    "source_stamp_labels": str(STAMP_LABELS_DIR),
    "source_index": str(INDEX_PATH),
    "source_state_merged": str(STATE_PATH),
    "classes": CLASSES
}, ensure_ascii=False, indent=2))

print(f"pages_written={pages_written}")
print(f"boxes_written={boxes_written}")
print(f"out_dir={OUT_DIR}")

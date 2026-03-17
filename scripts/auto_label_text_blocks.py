#!/usr/bin/env python3
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2


OCR_INSTANCE = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-label text lines or text blocks from page images using PaddleOCR."
    )
    p.add_argument("--images", default="/data/out/text_pages/images", help="Input images directory")
    p.add_argument("--out", default="/data/out/text_pages", help="Workspace output directory")
    p.add_argument("--lang", default="es", help="PaddleOCR language")
    p.add_argument(
        "--mode",
        choices=["line", "block"],
        default="block",
        help="Output one box per OCR line or merge lines into text blocks",
    )
    p.add_argument("--min-conf", type=float, default=0.4, help="Minimum OCR confidence")
    p.add_argument(
        "--merge-y-gap",
        type=int,
        default=22,
        help="Max vertical gap in pixels between nearby lines to merge into one block",
    )
    p.add_argument(
        "--merge-x-tolerance",
        type=int,
        default=80,
        help="Horizontal tolerance when deciding whether nearby lines belong to the same block",
    )
    p.add_argument("--padding", type=int, default=4, help="Extra padding around generated boxes")
    p.add_argument("--limit", type=int, default=0, help="Max number of images to process (0=all)")
    p.add_argument("--preview", action="store_true", help="Generate preview images")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="Number of worker processes for CPU PaddleOCR",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already have labels_auto; rebuild preview only if needed",
    )
    p.add_argument("--overwrite", action="store_true", help="Reprocess even when labels already exist")
    return p.parse_args()


def rect_from_quad(quad) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def clamp_box(box: tuple[float, float, float, float], width: int, height: int, padding: int):
    x0, y0, x1, y1 = box
    x0 = max(0, int(round(x0)) - padding)
    y0 = max(0, int(round(y0)) - padding)
    x1 = min(width, int(round(x1)) + padding)
    y1 = min(height, int(round(y1)) + padding)
    return x0, y0, x1, y1


def boxes_overlap_or_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    y_gap: int,
    x_tolerance: int,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    vertical_gap = max(0, max(by0 - ay1, ay0 - by1))
    if vertical_gap > y_gap:
        return False
    horizontal_overlap = min(ax1, bx1) - max(ax0, bx0)
    return horizontal_overlap >= -x_tolerance


def merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    y_gap: int,
    x_tolerance: int,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        last = merged[-1]
        if boxes_overlap_or_close(last, box, y_gap=y_gap, x_tolerance=x_tolerance):
            merged[-1] = (
                min(last[0], box[0]),
                min(last[1], box[1]),
                max(last[2], box[2]),
                max(last[3], box[3]),
            )
        else:
            merged.append(box)
    return merged


def save_yolo_labels(
    image_shape: tuple[int, int, int],
    boxes: list[tuple[int, int, int, int]],
    label_path: Path,
) -> None:
    height, width = image_shape[:2]
    lines = []
    for x0, y0, x1, y1 in boxes:
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        cx = (x0 + x1) / 2.0 / width
        cy = (y0 + y1) / 2.0 / height
        nw = bw / float(width)
        nh = bh / float(height)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def save_preview(image, boxes: list[tuple[int, int, int, int]], path: Path) -> None:
    preview = image.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 140, 255), 2)
    cv2.imwrite(path.as_posix(), preview)


def load_yolo_boxes(image_shape: tuple[int, int, int], label_path: Path) -> list[tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        cx = float(cx) * width
        cy = float(cy) * height
        bw = float(w) * width
        bh = float(h) * height
        x0 = int(round(cx - bw / 2.0))
        y0 = int(round(cy - bh / 2.0))
        x1 = int(round(cx + bw / 2.0))
        y1 = int(round(cy + bh / 2.0))
        boxes.append((x0, y0, x1, y1))
    return boxes


def init_worker(lang: str) -> None:
    global OCR_INSTANCE
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise SystemExit(f"PaddleOCR not available: {exc}")
    OCR_INSTANCE = PaddleOCR(use_angle_cls=False, lang=lang)


def process_image(task: dict[str, Any]) -> dict[str, Any]:
    global OCR_INSTANCE
    image_path = Path(task["image_path"])
    label_path = Path(task["label_path"])
    preview_path = Path(task["preview_path"])
    image = cv2.imread(str(image_path))
    if image is None:
        return {"image": image_path.name, "status": "error", "boxes": 0}

    if task["preview_only"]:
        boxes = load_yolo_boxes(image.shape, label_path)
        save_preview(image, boxes, preview_path)
        return {
            "image": image_path.name,
            "status": "preview_only",
            "ocr_lines": None,
            "boxes": len(boxes),
            "label_path": str(label_path),
        }

    result = OCR_INSTANCE.ocr(image, cls=False)
    line_boxes: list[tuple[int, int, int, int]] = []
    kept_lines = 0
    if result and result[0]:
        for item in result[0]:
            quad, payload = item
            text = ""
            conf = 1.0
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                text = str(payload[0] or "")
                conf = float(payload[1] or 0.0)
            if conf < task["min_conf"] or not text.strip():
                continue
            rect = rect_from_quad(quad)
            line_boxes.append(clamp_box(rect, image.shape[1], image.shape[0], task["padding"]))
            kept_lines += 1

    if task["mode"] == "block":
        boxes = merge_boxes(
            line_boxes,
            y_gap=task["merge_y_gap"],
            x_tolerance=task["merge_x_tolerance"],
        )
    else:
        boxes = sorted(line_boxes, key=lambda b: (b[1], b[0]))

    save_yolo_labels(image.shape, boxes, label_path)
    if task["preview"]:
        save_preview(image, boxes, preview_path)

    return {
        "image": image_path.name,
        "status": "processed",
        "ocr_lines": kept_lines,
        "boxes": len(boxes),
        "label_path": str(label_path),
    }


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")

    labels_auto_dir = Path(args.out) / "labels_auto"
    previews_dir = Path(args.out) / "previews_auto"
    labels_auto_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        previews_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.limit > 0:
        images = images[: args.limit]

    tasks = []
    skipped = 0
    preview_only = 0
    for image_path in images:
        label_path = labels_auto_dir / f"{image_path.stem}.txt"
        preview_path = previews_dir / image_path.name
        if args.skip_existing and not args.overwrite and label_path.exists():
            if args.preview and not preview_path.exists():
                tasks.append(
                    {
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "preview_path": str(preview_path),
                        "mode": args.mode,
                        "min_conf": args.min_conf,
                        "merge_y_gap": args.merge_y_gap,
                        "merge_x_tolerance": args.merge_x_tolerance,
                        "padding": args.padding,
                        "preview": True,
                        "preview_only": True,
                    }
                )
                preview_only += 1
            else:
                skipped += 1
            continue

        tasks.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "preview_path": str(preview_path),
                "mode": args.mode,
                "min_conf": args.min_conf,
                "merge_y_gap": args.merge_y_gap,
                "merge_x_tolerance": args.merge_x_tolerance,
                "padding": args.padding,
                "preview": args.preview,
                "preview_only": False,
            }
        )

    print(f"Total images: {len(images)}")
    print(f"Skipped existing labels: {skipped}")
    print(f"Preview-only jobs: {preview_only}")
    print(f"OCR jobs to run: {sum(1 for t in tasks if not t['preview_only'])}")
    print(f"Workers: {args.workers}")

    if not tasks:
        print("Nothing to do.")
        return 0

    report_rows = []
    completed = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_worker,
        initargs=(args.lang,),
    ) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            report_rows.append(
                {
                    "image": row["image"],
                    "ocr_lines": row.get("ocr_lines"),
                    "boxes": row.get("boxes"),
                    "mode": args.mode,
                    "label_path": row.get("label_path"),
                    "status": row.get("status"),
                }
            )
            print(
                f"[{completed}/{len(tasks)}] {row['image']}: "
                f"status={row.get('status')} boxes={row.get('boxes')}"
            )

    report_path = Path(args.out) / "auto_labels_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "ocr_lines", "boxes", "mode", "label_path", "status"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Auto labels: {labels_auto_dir}")
    if args.preview:
        print(f"Previews: {previews_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

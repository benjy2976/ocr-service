#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import fitz
import cv2
import numpy as np


def render_page(pdf_path: Path, page_index: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img
    finally:
        doc.close()


def find_candidates(
    img_bgr: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
    aspect_min: float,
    aspect_max: float,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * (area / (peri * peri))
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0.0
        if circularity >= min_circularity or (aspect_min <= aspect <= aspect_max):
            boxes.append((x, y, w, h))
    return boxes


def save_preview(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], path: Path) -> None:
    preview = img_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(path.as_posix(), preview)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stamp candidates from PDFs.")
    parser.add_argument("--list", default="data/sample_list.txt", help="List of PDF paths")
    parser.add_argument("--out", default="data/out/stamp_candidates", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    parser.add_argument("--max-pages", type=int, default=1, help="Max pages per PDF")
    parser.add_argument("--min-area", type=float, default=2000, help="Min contour area")
    parser.add_argument("--max-area", type=float, default=200000, help="Max contour area")
    parser.add_argument("--min-circularity", type=float, default=0.4, help="Min circularity")
    parser.add_argument("--aspect-min", type=float, default=0.5, help="Min aspect ratio")
    parser.add_argument("--aspect-max", type=float, default=2.0, help="Max aspect ratio")
    parser.add_argument("--preview", action="store_true", help="Save preview images with boxes")
    args = parser.parse_args()

    list_path = Path(args.list)
    if not list_path.exists():
        print(f"List not found: {list_path}")
        return 1

    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    previews_dir = out_dir / "previews"
    crops_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "candidates.csv"
    rows = []

    pdfs = [p.strip() for p in list_path.read_text().splitlines() if p.strip()]
    for p in pdfs:
        if p.startswith("data/"):
            pdf_path = Path("/" + p)
        else:
            pdf_path = Path(p)
        if not pdf_path.is_absolute():
            pdf_path = Path("/data") / pdf_path
        if not pdf_path.exists():
            print(f"Missing: {pdf_path}")
            continue

        doc = fitz.open(pdf_path)
        page_count = min(doc.page_count, args.max_pages)
        doc.close()

        for page_index in range(page_count):
            img = render_page(pdf_path, page_index, args.dpi)
            boxes = find_candidates(
                img,
                min_area=args.min_area,
                max_area=args.max_area,
                min_circularity=args.min_circularity,
                aspect_min=args.aspect_min,
                aspect_max=args.aspect_max,
            )

            if args.preview:
                preview_path = previews_dir / f"{pdf_path.stem}_p{page_index+1}.png"
                save_preview(img, boxes, preview_path)

            for i, (x, y, w, h) in enumerate(boxes, start=1):
                crop = img[y : y + h, x : x + w]
                crop_name = f"{pdf_path.stem}_p{page_index+1}_{i}.png"
                crop_path = crops_dir / crop_name
                cv2.imwrite(crop_path.as_posix(), crop)
                rows.append(
                    {
                        "pdf": str(pdf_path),
                        "page": page_index + 1,
                        "crop": str(crop_path),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdf", "page", "crop", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidates: {len(rows)}")
    print(f"Crops: {crops_dir}")
    print(f"CSV: {csv_path}")
    if args.preview:
        print(f"Previews: {previews_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

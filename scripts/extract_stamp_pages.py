#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import fitz
import cv2
import numpy as np


def render_page(pdf_path: Path, page_index: int, dpi: int) -> np.ndarray | None:
    try:
        fitz.TOOLS.set_verbosity(0)
    except Exception:
        pass
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img
    except Exception:
        return None
    finally:
        doc.close()


def find_candidates(
    img_bgr: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
    aspect_min: float,
    aspect_max: float,
    max_boxes: int,
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

    # Keep largest boxes first
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_boxes]


def save_preview(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], path: Path) -> None:
    preview = img_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(path.as_posix(), preview)


def save_yolo_labels(
    img_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    label_path: Path,
) -> None:
    h, w = img_shape[:2]
    lines = []
    for x, y, bw, bh in boxes:
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract full-page candidates and YOLO labels.")
    parser.add_argument("--list", default="/data/sample_list.txt", help="List of PDF paths")
    parser.add_argument("--out", default="/data/out/stamp_pages", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    parser.add_argument("--max-pages", type=int, default=1, help="Max pages per PDF (0=all)")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--log-errors", action="store_true", help="Save errors to CSV")
    parser.add_argument("--min-area", type=float, default=3000, help="Min contour area")
    parser.add_argument("--max-area", type=float, default=400000, help="Max contour area")
    parser.add_argument("--min-circularity", type=float, default=0.45, help="Min circularity")
    parser.add_argument("--aspect-min", type=float, default=0.5, help="Min aspect ratio")
    parser.add_argument("--aspect-max", type=float, default=2.0, help="Max aspect ratio")
    parser.add_argument("--max-boxes", type=int, default=8, help="Max boxes per page")
    args = parser.parse_args()

    list_path = Path(args.list)
    if not list_path.exists():
        print(f"List not found: {list_path}")
        return 1

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    previews_dir = out_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pages.csv"
    rows = []
    errors = []

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
            if args.log_errors:
                errors.append(
                    {
                        "pdf": str(pdf_path),
                        "page": "",
                        "error": "missing",
                    }
                )
            continue

        try:
            doc = fitz.open(pdf_path)
            if args.max_pages == 0:
                page_count = doc.page_count
            else:
                page_count = min(doc.page_count, args.max_pages)
            doc.close()
        except Exception:
            if args.verbose:
                print(f"  Skip PDF: open failed {pdf_path.name}", flush=True)
            if args.log_errors:
                errors.append(
                    {
                        "pdf": str(pdf_path),
                        "page": "",
                        "error": "open_failed",
                    }
                )
            continue

        for page_index in range(page_count):
            if args.verbose:
                print(f"PDF {pdf_path.name} page {page_index+1}/{page_count}", flush=True)
            img = render_page(pdf_path, page_index, args.dpi)
            if img is None:
                if args.verbose:
                    print(f"  Skip page {page_index+1}: render failed", flush=True)
                if args.log_errors:
                    errors.append(
                        {
                            "pdf": str(pdf_path),
                            "page": page_index + 1,
                            "error": "render_failed",
                        }
                    )
                continue
            boxes = find_candidates(
                img,
                min_area=args.min_area,
                max_area=args.max_area,
                min_circularity=args.min_circularity,
                aspect_min=args.aspect_min,
                aspect_max=args.aspect_max,
                max_boxes=args.max_boxes,
            )

            img_name = f"{pdf_path.stem}_p{page_index+1}.png"
            img_path = images_dir / img_name
            cv2.imwrite(img_path.as_posix(), img)

            label_path = labels_dir / (Path(img_name).stem + ".txt")
            save_yolo_labels(img.shape, boxes, label_path)

            preview_path = previews_dir / img_name
            save_preview(img, boxes, preview_path)

            rows.append(
                {
                    "pdf": str(pdf_path),
                    "page": page_index + 1,
                    "image": str(img_path),
                    "labels": str(label_path),
                    "preview": str(preview_path),
                    "boxes": len(boxes),
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdf", "page", "image", "labels", "preview", "boxes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.log_errors and errors:
        err_path = out_dir / "errors.csv"
        with err_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pdf", "page", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"Errors: {err_path}")

    print(f"Pages: {len(rows)}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Previews: {previews_dir}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

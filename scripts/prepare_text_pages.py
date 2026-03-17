#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare a separate text-detection workspace from existing stamp pages."
    )
    p.add_argument("--src", default="/data/out/stamp_pages", help="Source stamp_pages directory")
    p.add_argument("--out", default="/data/out/text_pages", help="Output text_pages directory")
    p.add_argument(
        "--link",
        action="store_true",
        help="Use hardlinks when possible instead of copying image files",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files in destination",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, *, link: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        if dst.is_file():
            dst.unlink()
    if link:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    src_images = src_dir / "images"
    src_pages_csv = src_dir / "pages.csv"
    if not src_images.exists():
        raise SystemExit(f"Missing images directory: {src_images}")

    out_images = out_dir / "images"
    out_labels_auto = out_dir / "labels_auto"
    out_labels_reviewed = out_dir / "labels_reviewed"
    out_previews = out_dir / "previews_auto"
    for path in (out_images, out_labels_auto, out_labels_reviewed, out_previews):
        ensure_dir(path)

    copied = 0
    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        dst = out_images / img_path.name
        copy_or_link(img_path, dst, link=args.link, overwrite=args.overwrite)
        copied += 1

    manifest_path = out_dir / "pages.csv"
    if src_pages_csv.exists():
        rows = []
        with src_pages_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = Path(row.get("image", "")).name
                rows.append(
                    {
                        "pdf": row.get("pdf", ""),
                        "page": row.get("page", ""),
                        "image": str(out_images / image_name) if image_name else "",
                        "source_image": row.get("image", ""),
                    }
                )
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pdf", "page", "image", "source_image"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Prepared text pages in: {out_dir}")
    print(f"Images: {copied}")
    print(f"Images dir: {out_images}")
    print(f"Auto labels dir: {out_labels_auto}")
    print(f"Reviewed labels dir: {out_labels_reviewed}")
    if manifest_path.exists():
        print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

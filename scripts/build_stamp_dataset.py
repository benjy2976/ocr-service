#!/usr/bin/env python3
from pathlib import Path
import random


def main() -> int:
    keep_dir = Path("/data/out/stamp_candidates/keep")
    reject_dir = Path("/data/out/stamp_candidates/reject")

    base = Path("/data/datasets/stamps_yolo")
    img_train = base / "images/train"
    img_val = base / "images/val"
    lbl_train = base / "labels/train"
    lbl_val = base / "labels/val"

    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    keep = [p for p in keep_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    reject = [p for p in reject_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    random.shuffle(keep)
    random.shuffle(reject)

    def split(lst, ratio=0.8):
        k = int(len(lst) * ratio)
        return lst[:k], lst[k:]

    keep_train, keep_val = split(keep)
    reject_train, reject_val = split(reject)

    def copy_img(src: Path, dst_dir: Path) -> None:
        (dst_dir / src.name).write_bytes(src.read_bytes())

    def add_positive(img_path: Path, img_out: Path, lbl_out: Path) -> None:
        copy_img(img_path, img_out)
        (lbl_out / (img_path.stem + ".txt")).write_text("0 0.5 0.5 1.0 1.0\n")

    def add_negative(img_path: Path, img_out: Path, lbl_out: Path) -> None:
        copy_img(img_path, img_out)
        (lbl_out / (img_path.stem + ".txt")).write_text("")

    for p in keep_train:
        add_positive(p, img_train, lbl_train)
    for p in keep_val:
        add_positive(p, img_val, lbl_val)
    for p in reject_train:
        add_negative(p, img_train, lbl_train)
    for p in reject_val:
        add_negative(p, img_val, lbl_val)

    (base / "data.yaml").write_text(
        f"path: {base.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['stamp']\n"
    )

    print("OK dataset:", base)
    print("keep:", len(keep), "reject:", len(reject))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

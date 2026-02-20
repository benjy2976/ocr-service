#!/usr/bin/env python3
import sys
from pathlib import Path

import fitz


def render_first_page(pdf_path: Path, out_dir: Path, dpi: int = 200) -> Path:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_p1.png"
        pix.save(out_path.as_posix())
        return out_path
    finally:
        doc.close()


def main() -> int:
    list_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_list.txt")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/out/previews")
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    if not list_file.exists():
        print(f"List not found: {list_file}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p.strip() for p in list_file.read_text().splitlines() if p.strip()]
    if not paths:
        print("List is empty.")
        return 1

    for p in paths:
        pdf_path = Path(p)
        if not pdf_path.exists():
            print(f"Missing: {pdf_path}")
            continue
        out_path = render_first_page(pdf_path, out_dir, dpi=dpi)
        print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

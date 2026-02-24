#!/usr/bin/env python3
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_file(root: Path, rel: str) -> Path:
    rel = rel.lstrip("/")
    return root / rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diverse train/test lists from regulations CSVs.")
    parser.add_argument("--regulations", default="data/samples/regulations.csv")
    parser.add_argument("--reg-files", default="data/samples/regulation_files.csv")
    parser.add_argument("--root", default="data/samples", help="Root where file_ruta lives")
    parser.add_argument("--url-base", default="", help="Base URL for downloads, e.g. https://.../regulations/file")
    parser.add_argument("--allow-missing", action="store_true", help="Allow entries without local files")
    parser.add_argument("--total", type=int, default=2000, help="Total sample size")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--min-per-year", type=int, default=5, help="Min samples per year")
    parser.add_argument("--min-per-type", type=int, default=5, help="Min samples per regulations_tipo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-train", default="/data/train_list.txt")
    parser.add_argument("--out-test", default="/data/test_list.txt")
    args = parser.parse_args()

    regs = read_csv(Path(args.regulations))
    reg_files = read_csv(Path(args.reg_files))

    reg_meta = {}
    for r in regs:
        reg_id = r.get("id")
        if not reg_id:
            continue
        reg_meta[reg_id] = {
            "year": r.get("reg_year") or "",
            "type": r.get("regulations_tipo") or "",
        }

    root = Path(args.root)
    samples = []
    for rf in reg_files:
        reg_id = rf.get("file_idregulation")
        ruta = rf.get("file_ruta") or ""
        if not reg_id or not ruta:
            continue
        meta = reg_meta.get(reg_id)
        if not meta:
            continue
        full = resolve_file(root, ruta)
        if not full.exists() and not args.allow_missing:
            continue
        url = ""
        if args.url_base:
            tomo = rf.get("file_tomo") or "0"
            nro_tipo = rf.get("file_nro_tipo") or "0"
            url = f"{args.url_base.rstrip('/')}/{reg_id}/{tomo}/{nro_tipo}"
        samples.append(
            {
                "path": str(full),
                "url": url,
                "year": meta["year"],
                "type": meta["type"],
            }
        )

    if not samples:
        print("No samples found. Check --root and CSVs.")
        return 1

    random.seed(args.seed)
    by_year = defaultdict(list)
    by_type = defaultdict(list)
    for s in samples:
        by_year[s["year"]].append(s)
        by_type[s["type"]].append(s)

    chosen = []
    used = set()

    def pick(lst, n):
        random.shuffle(lst)
        for item in lst:
            if len(chosen) >= n:
                break
            if item["path"] in used:
                continue
            used.add(item["path"])
            chosen.append(item)

    # Ensure per-year coverage
    for year, lst in by_year.items():
        pick(lst, args.min_per_year)

    # Ensure per-type coverage
    for typ, lst in by_type.items():
        pick(lst, args.min_per_type)

    # Fill remaining randomly
    remaining = [s for s in samples if s["path"] not in used]
    random.shuffle(remaining)
    for s in remaining:
        if len(chosen) >= args.total:
            break
        used.add(s["path"])
        chosen.append(s)

    random.shuffle(chosen)
    cut = int(len(chosen) * (1 - args.test_ratio))
    train = chosen[:cut]
    test = chosen[cut:]

    def format_line(s):
        return s["url"] if s["url"] else s["path"]

    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    out_train.write_text("\n".join(format_line(s) for s in train) + "\n")
    out_test.write_text("\n".join(format_line(s) for s in test) + "\n")

    print(f"Samples available: {len(samples)}")
    print(f"Chosen: {len(chosen)}")
    print(f"Train: {len(train)} -> {args.out_train}")
    print(f"Test: {len(test)} -> {args.out_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

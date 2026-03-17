#!/usr/bin/env python3
import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build classification dataset from labels.")
    p.add_argument("--labels", default="/data/classify/state.json", help="State JSON with labels")
    p.add_argument("--crops", default="/data/classify/crops", help="Crops dir")
    p.add_argument("--out", default="/data/classify/dataset", help="Dataset output dir")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--min-per-class", type=int, default=50, help="Minimum samples per class to include")
    p.add_argument("--force-include", default="", help="Comma-separated classes to include even if below min")
    p.add_argument("--report", default="/data/classify/reports/classify_report.json", help="Report output JSON")
    p.add_argument("--index", default="/data/classify/index.csv", help="Index CSV with source_image column")
    p.add_argument(
        "--split-by",
        choices=["random", "source_image"],
        default="source_image",
        help="Split strategy",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    state_path = Path(args.labels)
    if not state_path.exists():
        raise SystemExit(f"Missing labels: {state_path}")
    state = json.loads(state_path.read_text())
    items = state.get("items", {})
    labeled = [
        (k, v.get("label"))
        for k, v in items.items()
        if v.get("label") and v.get("label") not in {"__skipped__", "__rejected__"}
    ]
    if not labeled:
        raise SystemExit("No labeled items found.")

    force_include = {c.strip() for c in args.force_include.split(",") if c.strip()}
    per_class = {}
    for _, label in labeled:
        per_class[label] = per_class.get(label, 0) + 1
    included = {c for c, n in per_class.items() if n >= args.min_per_class}
    included |= force_include
    excluded = {c: n for c, n in per_class.items() if c not in included}

    filtered = [(name, label) for name, label in labeled if label in included]
    if not filtered:
        raise SystemExit("No labeled items after filtering.")

    random.seed(args.seed)
    train = []
    val = []
    if args.split_by == "source_image":
        index_path = Path(args.index)
        if not index_path.exists():
            raise SystemExit(f"Missing index: {index_path}")
        crop_to_source = {}
        with index_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop = row.get("crop")
                src = row.get("source_image")
                if crop and src:
                    crop_to_source[crop] = src
        groups: dict[str, list[tuple[str, str]]] = {}
        for name, label in filtered:
            src = crop_to_source.get(name, name)
            groups.setdefault(src, []).append((name, label))
        sources = list(groups.keys())
        random.shuffle(sources)
        val_count = max(1, int(len(sources) * args.val_ratio))
        val_sources = set(sources[:val_count])
        for src, rows in groups.items():
            if src in val_sources:
                val.extend(rows)
            else:
                train.extend(rows)
    else:
        # Stratified split per class to keep all classes in val
        by_class: dict[str, list[tuple[str, str]]] = {}
        for name, label in filtered:
            by_class.setdefault(label, []).append((name, label))
        for label, rows in by_class.items():
            random.shuffle(rows)
            n_val = max(1, int(len(rows) * args.val_ratio))
            val.extend(rows[:n_val])
            train.extend(rows[n_val:])

    out_dir = Path(args.out)
    for split_name, rows in (("train", train), ("val", val)):
        for name, label in rows:
            dest_dir = out_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            src = Path(args.crops) / name
            if not src.exists():
                continue
            shutil.copy2(src, dest_dir / name)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "min_per_class": args.min_per_class,
        "force_include": sorted(force_include),
        "included_classes": sorted(included),
        "excluded_classes": excluded,
        "counts": per_class,
        "train": len(train),
        "val": len(val),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Out: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())

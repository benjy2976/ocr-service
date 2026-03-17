#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Predict crop classes with a YOLO classifier.")
    p.add_argument("--model", required=True, help="Classifier model path (.pt)")
    p.add_argument("--crops", default="/data/classify/crops", help="Crops dir")
    p.add_argument("--out", default="/data/classify/preds.json", help="Output predictions JSON")
    p.add_argument("--conf", type=float, default=0.0, help="Confidence threshold")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    crops = sorted(Path(args.crops).glob("*.*"))
    preds = {}
    for path in crops:
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        result = model.predict(source=str(path), verbose=False)[0]
        if result.probs is None:
            continue
        top1 = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        if conf < args.conf:
            continue
        label = result.names.get(top1, str(top1))
        preds[path.name] = {"label": label, "confidence": conf}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"Preds: {len(preds)} -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())

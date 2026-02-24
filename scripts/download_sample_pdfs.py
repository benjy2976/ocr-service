#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def safe_name(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "file.pdf"
    name = path.replace("/", "_")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PDFs from list of URLs.")
    parser.add_argument("--list", default="/data/train_list.txt", help="URL list file")
    parser.add_argument("--out", default="/data/samples_train", help="Output directory")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between downloads")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of downloads (0=all)")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress")
    args = parser.parse_args()

    list_path = Path(args.list)
    if not list_path.exists():
        print(f"List not found: {list_path}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = [u.strip() for u in list_path.read_text().splitlines() if u.strip()]
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    session = requests.Session()
    ok = 0
    fail = 0

    for i, url in enumerate(urls, start=1):
        name = safe_name(url)
        dst = out_dir / name
        if dst.exists():
            if args.verbose:
                print(f"[{i}/{len(urls)}] exists {dst.name}")
            continue

        success = False
        for attempt in range(1, args.retries + 2):
            try:
                with session.get(url, stream=True, timeout=args.timeout) as resp:
                    resp.raise_for_status()
                    content_type = (resp.headers.get("content-type") or "").lower()
                    if "pdf" not in content_type:
                        raise RuntimeError(f"not a PDF (content-type={content_type})")
                    if args.verbose:
                        print(f"[{i}/{len(urls)}] download {url}", flush=True)
                    with dst.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                # verify PDF signature
                with dst.open("rb") as f:
                    if f.read(5) != b"%PDF-":
                        raise RuntimeError("invalid PDF header")
                success = True
                break
            except Exception as exc:
                if attempt <= args.retries:
                    print(f"[{i}/{len(urls)}] retry {attempt} {url} ({exc})")
                    time.sleep(1)
                else:
                    print(f"[{i}/{len(urls)}] fail {url} ({exc})")

        if success:
            ok += 1
        else:
            fail += 1
            if dst.exists():
                dst.unlink(missing_ok=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. ok={ok} fail={fail} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

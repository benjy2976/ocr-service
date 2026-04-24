from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from opensearchpy import helpers

from app.search_common import (
    OCR_SHARED_CACHE_DIR,
    OPENSEARCH_INDEX,
    UnsupportedTextArtifact,
    ensure_index,
    iter_page_documents,
    opensearch_client,
)


logging.basicConfig(
    level=os.getenv("SEARCH_INDEXER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] search-indexer - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("search-indexer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index OCR text JSON artifacts into OpenSearch")
    parser.add_argument("--cache-root", default=str(OCR_SHARED_CACHE_DIR))
    parser.add_argument("--index", default=OPENSEARCH_INDEX)
    parser.add_argument("--path", help="Index one text JSON file or one directory")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.getenv("SEARCH_INDEXER_POLL_INTERVAL_SECONDS", "300")),
        help="Seconds between scans when not using --once",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("SEARCH_INDEXER_BATCH_SIZE", "500")),
    )
    args = parser.parse_args()

    client = opensearch_client()
    ensure_index(client, args.index)

    cache_root = Path(args.cache_root)
    target = Path(args.path) if args.path else cache_root

    while True:
        stats = index_target(
            target,
            cache_root=cache_root,
            index_name=args.index,
            batch_size=args.batch_size,
            client=client,
        )
        logger.info(
            "scan complete | files=%d indexed_pages=%d skipped=%d errors=%d",
            stats["files"],
            stats["indexed_pages"],
            stats["skipped"],
            stats["errors"],
        )
        if args.once:
            return
        time.sleep(args.sleep)


def index_target(
    target: Path,
    *,
    cache_root: Path,
    index_name: str,
    batch_size: int,
    client,
) -> dict[str, int]:
    stats = {"files": 0, "indexed_pages": 0, "skipped": 0, "errors": 0}
    actions = []

    for artifact_path in iter_text_artifacts(target):
        stats["files"] += 1
        try:
            page_count = 0
            for doc_id, doc in iter_page_documents(artifact_path, cache_root=cache_root):
                actions.append({
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": doc,
                })
                page_count += 1
                if len(actions) >= batch_size:
                    stats["indexed_pages"] += flush_actions(client, actions)
                    actions.clear()
            if page_count == 0:
                stats["skipped"] += 1
        except UnsupportedTextArtifact:
            stats["skipped"] += 1
        except Exception as exc:
            stats["errors"] += 1
            logger.warning("failed to index %s: %s", artifact_path, exc)

    if actions:
        stats["indexed_pages"] += flush_actions(client, actions)
    return stats


def iter_text_artifacts(target: Path):
    if target.is_file():
        yield target
        return
    if not target.exists():
        logger.warning("target does not exist: %s", target)
        return
    yield from target.rglob("*.jsonl")


def flush_actions(client, actions: list[dict]) -> int:
    success, _errors = helpers.bulk(client, actions, raise_on_error=False)
    return int(success)


if __name__ == "__main__":
    main()

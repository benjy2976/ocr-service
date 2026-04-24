from __future__ import annotations

from typing import Literal
from typing import Any

from fastapi import FastAPI, Query

from app.search_common import OPENSEARCH_INDEX, ensure_index, opensearch_client


app = FastAPI(title="OCR Search API", version="0.1.0")


@app.get("/health")
def health():
    client = opensearch_client()
    return {
        "status": "ok",
        "opensearch": client.ping(),
        "index": OPENSEARCH_INDEX,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    regulation_file_id: int | None = None,
    regulation_id: int | None = None,
    year: int | None = None,
    tipo: int | None = None,
    sigla_id: int | None = None,
    regulation_type_id: int | None = None,
    regulation_type_sigla_id: int | None = None,
    group_by: Literal["regulation", "file", "page", "document"] = "regulation",
    matched_files_limit: int = Query(10, ge=1, le=50),
    matched_pages_limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    client = opensearch_client()
    ensure_index(client, OPENSEARCH_INDEX)

    filters = []
    if regulation_file_id is not None:
        filters.append({"term": {"regulation_file_id": regulation_file_id}})
    if regulation_id is not None:
        filters.append({"term": {"regulation_id": regulation_id}})
    if year is not None:
        filters.append({"term": {"reg_year": year}})
    if tipo is not None:
        filters.append({"term": {"regulations_tipo": tipo}})
    if regulation_type_id is not None:
        filters.append({"term": {"regulation_type_id": regulation_type_id}})
    if sigla_id is not None:
        filters.append({"term": {"regulation_type_sigla_id": sigla_id}})
    if regulation_type_sigla_id is not None:
        filters.append({"term": {"regulation_type_sigla_id": regulation_type_sigla_id}})

    effective_group_by = "file" if group_by == "document" else group_by
    body = {
        "from": offset,
        "size": limit,
        "_source": [
            "regulation_file_id",
            "regulation_id",
            "source_md5",
            "page",
            "page_count",
            "text_path",
            "pdf_path",
            "source_path",
            "file_name",
            "reg_num",
            "reg_year",
            "reg_date",
            "reg_title",
            "reg_description",
            "regulations_tipo",
            "regulations_tipos_sigla_id",
            "regulation_type_id",
            "regulation_type_sigla_id",
            "text_source_kind",
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": q,
                            "fields": [
                                "text^4",
                                "reg_title^3",
                                "reg_description^2",
                                "file_name",
                            ],
                            "operator": "and",
                        }
                    }
                ],
                "filter": filters,
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "text": {"fragment_size": 180, "number_of_fragments": 3},
                "reg_title": {"fragment_size": 120, "number_of_fragments": 1},
                "reg_description": {"fragment_size": 180, "number_of_fragments": 2},
            },
        },
        "aggs": {
            "unique_regulations": {
                "cardinality": {
                    "field": "regulation_id",
                    "precision_threshold": 40000,
                }
            },
            "unique_files": {
                "cardinality": {
                    "field": "regulation_file_id",
                    "precision_threshold": 40000,
                }
            }
        },
    }
    if effective_group_by == "regulation":
        body["collapse"] = {
            "field": "regulation_id",
            "inner_hits": {
                "name": "matched_pages",
                "size": matched_files_limit * matched_pages_limit,
                "_source": [
                    "regulation_file_id",
                    "file_name",
                    "pdf_path",
                    "text_path",
                    "source_path",
                    "page",
                    "char_count",
                    "word_count",
                ],
                "highlight": {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fields": {
                        "text": {"fragment_size": 180, "number_of_fragments": 3},
                    },
                },
                "sort": [{"_score": "desc"}],
            },
        }
    elif effective_group_by == "file":
        body["collapse"] = {
            "field": "regulation_file_id",
            "inner_hits": {
                "name": "matched_pages",
                "size": matched_pages_limit,
                "_source": ["page", "char_count", "word_count"],
                "highlight": {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fields": {
                        "text": {"fragment_size": 180, "number_of_fragments": 3},
                    },
                },
                "sort": [{"_score": "desc"}],
            },
        }

    response = client.search(index=OPENSEARCH_INDEX, body=body)
    hits = response.get("hits", {})
    aggregations = response.get("aggregations") or {}
    unique_regulations = aggregations.get("unique_regulations") or {}
    unique_files = aggregations.get("unique_files") or {}
    return {
        "query": q,
        "group_by": effective_group_by,
        "total": (
            int(unique_regulations.get("value") or 0)
            if effective_group_by == "regulation"
            else int(unique_files.get("value") or 0)
            if effective_group_by == "file"
            else _total_value(hits.get("total"))
        ),
        "total_page_matches": _total_value(hits.get("total")),
        "limit": limit,
        "offset": offset,
        "results": [
            _format_hit(
                hit,
                group_by=effective_group_by,
                matched_pages_limit=matched_pages_limit,
            )
            for hit in hits.get("hits", [])
        ],
    }


def _total_value(total: Any) -> int:
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    if total is None:
        return 0
    return int(total)


def _format_hit(
    hit: dict[str, Any],
    *,
    group_by: str,
    matched_pages_limit: int,
) -> dict[str, Any]:
    result = {
        **hit.get("_source", {}),
        "score": hit.get("_score"),
        "highlight": hit.get("highlight", {}),
    }
    if group_by not in ("regulation", "file"):
        return result

    inner_hits = hit.get("inner_hits") or {}
    matched = inner_hits.get("matched_pages") or {}
    matched_pages = [
        {
            **page_hit.get("_source", {}),
            "score": page_hit.get("_score"),
            "highlight": page_hit.get("highlight", {}),
        }
        for page_hit in (matched.get("hits") or {}).get("hits", [])
    ]

    if group_by == "file":
        result["matched_pages"] = matched_pages[:matched_pages_limit]
        return result

    result["matched_files"] = _group_pages_by_file(
        matched_pages,
        matched_pages_limit=matched_pages_limit,
    )
    return result


def _group_pages_by_file(
    matched_pages: list[dict[str, Any]],
    *,
    matched_pages_limit: int,
) -> list[dict[str, Any]]:
    files: dict[Any, dict[str, Any]] = {}
    for page in matched_pages:
        file_id = page.get("regulation_file_id")
        if file_id not in files:
            files[file_id] = {
                "regulation_file_id": file_id,
                "file_name": page.get("file_name"),
                "pdf_path": page.get("pdf_path"),
                "text_path": page.get("text_path"),
                "source_path": page.get("source_path"),
                "score": page.get("score"),
                "matched_pages": [],
            }
        entry = files[file_id]
        if len(entry["matched_pages"]) >= matched_pages_limit:
            continue
        entry["matched_pages"].append({
            "page": page.get("page"),
            "char_count": page.get("char_count"),
            "word_count": page.get("word_count"),
            "score": page.get("score"),
            "highlight": page.get("highlight", {}),
        })
    return list(files.values())

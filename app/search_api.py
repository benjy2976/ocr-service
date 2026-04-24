from __future__ import annotations

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
    }

    response = client.search(index=OPENSEARCH_INDEX, body=body)
    hits = response.get("hits", {})
    return {
        "query": q,
        "total": _total_value(hits.get("total")),
        "limit": limit,
        "offset": offset,
        "results": [
            {
                **hit.get("_source", {}),
                "score": hit.get("_score"),
                "highlight": hit.get("highlight", {}),
            }
            for hit in hits.get("hits", [])
        ],
    }


def _total_value(total: Any) -> int:
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    if total is None:
        return 0
    return int(total)

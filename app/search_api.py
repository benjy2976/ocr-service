from __future__ import annotations

from typing import Literal
from typing import Any

from fastapi import FastAPI, HTTPException, Query

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
    q: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    offset: int | None = Query(None, ge=0),
    page: int | None = Query(None, ge=1),
    per_page: int | None = Query(None, ge=1, le=100),
    sort: str | None = Query(None, min_length=1),
    sort_by: str | None = Query(None, min_length=1),
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
    highlight_fragment_size: int = Query(180, ge=80, le=500),
    highlight_fragments: int = Query(3, ge=1, le=5),
) -> dict[str, Any]:
    client = opensearch_client()
    ensure_index(client, OPENSEARCH_INDEX)
    query_text = (q or "").strip()
    mode = "search" if query_text else "list"
    page_size, page_offset, current_page = _resolve_pagination(
        limit=limit,
        offset=offset,
        page=page,
        per_page=per_page,
    )
    default_sort = "-score" if mode == "search" else "-reg_date"
    sort_name, sort_clause = _resolve_sort(sort=sort, sort_by=sort_by, default=default_sort)

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
        "from": page_offset,
        "size": page_size,
        "sort": sort_clause,
        "track_scores": True,
        "track_total_hits": True,
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
        "query": _build_query(query_text, filters),
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
    if mode == "search":
        body["highlight"] = _highlight_config(
            fragment_size=highlight_fragment_size,
            fragments=highlight_fragments,
            include_metadata_fields=True,
        )
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
                "sort": [{"_score": "desc"}],
            },
        }
        if mode == "search":
            body["collapse"]["inner_hits"]["highlight"] = _highlight_config(
                fragment_size=highlight_fragment_size,
                fragments=highlight_fragments,
                include_metadata_fields=False,
            )
    elif effective_group_by == "file":
        body["collapse"] = {
            "field": "regulation_file_id",
            "inner_hits": {
                "name": "matched_pages",
                "size": matched_pages_limit,
                "_source": ["page", "char_count", "word_count"],
                "sort": [{"_score": "desc"}],
            },
        }
        if mode == "search":
            body["collapse"]["inner_hits"]["highlight"] = _highlight_config(
                fragment_size=highlight_fragment_size,
                fragments=highlight_fragments,
                include_metadata_fields=False,
            )

    response = client.search(index=OPENSEARCH_INDEX, body=body)
    hits = response.get("hits", {})
    aggregations = response.get("aggregations") or {}
    unique_regulations = aggregations.get("unique_regulations") or {}
    unique_files = aggregations.get("unique_files") or {}
    total = (
        int(unique_regulations.get("value") or 0)
        if effective_group_by == "regulation"
        else int(unique_files.get("value") or 0)
        if effective_group_by == "file"
        else _total_value(hits.get("total"))
    )
    results = [
        _format_hit(
            hit,
            group_by=effective_group_by,
            matched_pages_limit=matched_pages_limit,
        )
        for hit in hits.get("hits", [])
    ]
    pagination = _pagination_payload(
        total=total,
        page_size=page_size,
        page_offset=page_offset,
        current_page=current_page,
        result_count=len(results),
    )
    return {
        "mode": mode,
        "query": query_text or None,
        "group_by": effective_group_by,
        "sort": sort_name,
        "highlight_fragment_size": highlight_fragment_size,
        "highlight_fragments": highlight_fragments,
        "total": total,
        "total_page_matches": _total_value(hits.get("total")),
        "limit": page_size,
        "offset": page_offset,
        "per_page": page_size,
        "current_page": current_page,
        "last_page": pagination["last_page"],
        "from": pagination["from"],
        "to": pagination["to"],
        "next_page": pagination["next_page"],
        "prev_page": pagination["prev_page"],
        "pagination": pagination,
        "results": results,
    }


def _resolve_pagination(
    *,
    limit: int | None,
    offset: int | None,
    page: int | None,
    per_page: int | None,
) -> tuple[int, int, int]:
    page_size = per_page or limit or 10
    if page is not None:
        return page_size, (page - 1) * page_size, page

    page_offset = offset or 0
    current_page = (page_offset // page_size) + 1
    return page_size, page_offset, current_page


def _build_query(query_text: str, filters: list[dict[str, Any]]) -> dict[str, Any]:
    must: list[dict[str, Any]] = []
    if query_text:
        must.append({
            "multi_match": {
                "query": query_text,
                "fields": [
                    "text^4",
                    "reg_title^3",
                    "reg_description^2",
                    "file_name",
                ],
                "operator": "and",
            }
        })

    return {
        "bool": {
            "must": must or [{"match_all": {}}],
            "filter": filters,
        }
    }


def _highlight_config(
    *,
    fragment_size: int,
    fragments: int,
    include_metadata_fields: bool,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "text": {
            "fragment_size": fragment_size,
            "number_of_fragments": fragments,
        },
    }
    if include_metadata_fields:
        fields["reg_title"] = {"fragment_size": 120, "number_of_fragments": 1}
        fields["reg_description"] = {"fragment_size": 180, "number_of_fragments": 2}
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": fields,
    }


def _resolve_sort(
    *,
    sort: str | None,
    sort_by: str | None,
    default: str,
) -> tuple[str, list[dict[str, Any]]]:
    requested = (sort_by or sort or default).strip()
    aliases = {
        "relevance": "-score",
        "score_desc": "-score",
        "date": "-reg_date",
        "date_desc": "-reg_date",
        "date_asc": "reg_date",
        "year_desc": "-reg_year",
        "year_asc": "reg_year",
        "num_desc": "-reg_num",
        "num_asc": "reg_num",
    }
    requested = aliases.get(requested, requested)

    direction = "asc"
    field = requested
    if requested.startswith("-"):
        direction = "desc"
        field = requested[1:]

    allowed_fields = {
        "score": "_score",
        "_score": "_score",
        "reg_date": "reg_date",
        "reg_year": "reg_year",
        "reg_num": "reg_num",
    }
    opensearch_field = allowed_fields.get(field)
    if opensearch_field is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported sort. Use -score, score, -reg_date, reg_date, "
                "-reg_year, reg_year, -reg_num or reg_num."
            ),
        )

    canonical = f"-{field}" if direction == "desc" else field
    primary = (
        {"_score": {"order": direction}}
        if opensearch_field == "_score"
        else {opensearch_field: {"order": direction, "missing": "_last"}}
    )
    tie_breakers = [
        {"_score": {"order": "desc"}},
        {"reg_date": {"order": "desc", "missing": "_last"}},
        {"regulation_id": {"order": "desc", "missing": "_last"}},
        {"regulation_file_id": {"order": "desc", "missing": "_last"}},
        {"page": {"order": "asc", "missing": "_last"}},
    ]
    sort_clause = [primary]
    for tie_breaker in tie_breakers:
        if tie_breaker not in sort_clause:
            sort_clause.append(tie_breaker)
    return canonical, sort_clause


def _pagination_payload(
    *,
    total: int,
    page_size: int,
    page_offset: int,
    current_page: int,
    result_count: int,
) -> dict[str, int | None]:
    last_page = max(1, (total + page_size - 1) // page_size) if total else 1
    from_item = page_offset + 1 if result_count else 0
    to_item = page_offset + result_count if result_count else 0
    return {
        "total": total,
        "per_page": page_size,
        "current_page": current_page,
        "last_page": last_page,
        "from": from_item,
        "to": to_item,
        "next_page": current_page + 1 if current_page < last_page else None,
        "prev_page": current_page - 1 if current_page > 1 else None,
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

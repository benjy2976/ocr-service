from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from opensearchpy import OpenSearch


OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "munis_ocr_pages")
OCR_SHARED_CACHE_DIR = Path(os.getenv("OCR_SHARED_CACHE_DIR", "/nfs-cache"))
TEXT_SCHEMA = "ocr.text.document.v1"


class UnsupportedTextArtifact(ValueError):
    pass


def opensearch_client() -> OpenSearch:
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        timeout=int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("OPENSEARCH_MAX_RETRIES", "3")),
        retry_on_timeout=True,
    )


def ensure_index(client: OpenSearch, index_name: str = OPENSEARCH_INDEX) -> None:
    if client.indices.exists(index=index_name):
        return
    client.indices.create(
        index=index_name,
        body={
            "settings": {
                "index": {
                    "number_of_shards": int(os.getenv("OPENSEARCH_SHARDS", "1")),
                    "number_of_replicas": int(os.getenv("OPENSEARCH_REPLICAS", "0")),
                },
                "analysis": {
                    "analyzer": {
                        "spanish_text": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "spanish_stop",
                                "spanish_stemmer",
                            ],
                        }
                    },
                    "filter": {
                        "spanish_stop": {
                            "type": "stop",
                            "stopwords": "_spanish_",
                        },
                        "spanish_stemmer": {
                            "type": "stemmer",
                            "language": "light_spanish",
                        },
                    },
                },
            },
            "mappings": {
                "dynamic": "false",
                "properties": {
                    "regulation_file_id": {"type": "long"},
                    "regulation_id": {"type": "long"},
                    "source_md5": {"type": "keyword"},
                    "text_path": {"type": "keyword"},
                    "pdf_path": {"type": "keyword"},
                    "source_path": {"type": "keyword"},
                    "file_name": {
                        "type": "text",
                        "analyzer": "spanish_text",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "file_size": {"type": "keyword"},
                    "reg_num": {"type": "integer"},
                    "reg_year": {"type": "integer"},
                    "reg_date": {"type": "date", "ignore_malformed": True},
                    "reg_title": {
                        "type": "text",
                        "analyzer": "spanish_text",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "reg_description": {
                        "type": "text",
                        "analyzer": "spanish_text",
                    },
                    "regulations_tipo": {"type": "integer"},
                    "regulations_tipos_sigla_id": {"type": "integer"},
                    "regulation_type_id": {"type": "integer"},
                    "regulation_type_sigla_id": {"type": "integer"},
                    "page": {"type": "integer"},
                    "page_count": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "text_source_kind": {"type": "keyword"},
                    "extraction_engine": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "spanish_text",
                    },
                },
            },
        },
    )


def load_text_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != TEXT_SCHEMA:
        raise UnsupportedTextArtifact(f"Unsupported text artifact schema: {path}")
    if not isinstance(payload.get("pages"), list):
        raise ValueError(f"Text artifact without pages array: {path}")
    return payload


def iter_page_documents(
    artifact_path: Path,
    *,
    cache_root: Path = OCR_SHARED_CACHE_DIR,
) -> Iterable[tuple[str, dict[str, Any]]]:
    payload = load_text_artifact(artifact_path)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    relative_text_path = _relative_to_cache(artifact_path, cache_root)

    base_doc = {
        "regulation_file_id": _int_or_none(metadata.get("regulation_file_id")),
        "regulation_id": _int_or_none(metadata.get("regulation_id")),
        "source_md5": metadata.get("source_md5"),
        "text_path": metadata.get("text_path") or relative_text_path,
        "pdf_path": metadata.get("pdf_path"),
        "source_path": metadata.get("source_path"),
        "file_name": metadata.get("file_name"),
        "file_size": str(metadata.get("file_size")) if metadata.get("file_size") is not None else None,
        "reg_num": _int_or_none(metadata.get("reg_num")),
        "reg_year": _int_or_none(metadata.get("reg_year")),
        "reg_date": metadata.get("reg_date"),
        "reg_title": metadata.get("reg_title"),
        "reg_description": metadata.get("reg_description"),
        "regulations_tipo": _int_or_none(metadata.get("regulations_tipo")),
        "regulations_tipos_sigla_id": _int_or_none(metadata.get("regulations_tipos_sigla_id")),
        "regulation_type_id": _int_or_none(
            metadata.get("regulation_type_id") or metadata.get("regulations_tipo")
        ),
        "regulation_type_sigla_id": _int_or_none(
            metadata.get("regulation_type_sigla_id") or metadata.get("regulations_tipos_sigla_id")
        ),
        "page_count": _int_or_none(payload.get("page_count")),
        "text_source_kind": payload.get("text_source_kind"),
        "extraction_engine": payload.get("extraction_engine"),
    }

    for page in payload["pages"]:
        if not isinstance(page, dict) or page.get("empty"):
            continue
        text = str(page.get("text") or "").strip()
        if not text:
            continue
        page_number = _int_or_none(page.get("page"))
        doc = {
            **base_doc,
            "page": page_number,
            "char_count": _int_or_none(page.get("char_count")),
            "word_count": _int_or_none(page.get("word_count")),
            "text": text,
        }
        yield build_document_id(doc, relative_text_path), doc


def build_document_id(doc: dict[str, Any], relative_text_path: str) -> str:
    file_id = doc.get("regulation_file_id") or doc.get("source_md5") or relative_text_path
    return f"{file_id}:p{doc.get('page') or 0}"


def _relative_to_cache(path: Path, cache_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(cache_root.resolve()))
    except ValueError:
        return str(path)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

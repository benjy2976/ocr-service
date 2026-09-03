"""Construccion del artefacto TEXT (JSONL) a partir de un PDF ya resuelto.

Generico: no conoce Normatividad ni ningun otro dominio. El llamador puede
pasar un `metadata` arbitrario que se guarda tal cual en el documento; el
motor no le da ningun significado.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.ocr_pipeline import _extract_text_pages

TEXT_SCHEMA = "ocr.text.document.v1"
_WORD_RE = re.compile(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]{2,}\b", re.UNICODE)


def build_text_document(
    pdf_path: Path,
    *,
    text_source_kind: str,
    extraction_engine: str = "pymupdf",
    metadata: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Devuelve (contenido_jsonl_utf8, stats) para el PDF dado.

    `stats` trae page_count / non_empty_pages / text_len, para logging del
    llamador. Lanza RuntimeError si todas las paginas quedan vacias.
    """
    extracted_pages = _extract_text_pages(pdf_path)
    page_count = len(extracted_pages)
    pages: list[dict[str, Any]] = []
    non_empty_pages = 0
    total_text_len = 0

    for extracted_page in extracted_pages:
        page_number = int(extracted_page["page"])
        text = str(extracted_page.get("text") or "").strip()
        char_count = len(text)
        word_count = len(_WORD_RE.findall(text))
        empty = not bool(text)
        if not empty:
            non_empty_pages += 1
        total_text_len += char_count
        pages.append({
            "page": page_number,
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "empty": empty,
        })

    if non_empty_pages == 0:
        raise RuntimeError(f"TEXT JSON derivado vacio desde PDF base: {pdf_path}")

    payload = {
        "schema": TEXT_SCHEMA,
        "page_count": page_count,
        "non_empty_pages": non_empty_pages,
        "text_len": total_text_len,
        "text_source_kind": text_source_kind,
        "extraction_engine": extraction_engine,
        "metadata": metadata or {},
        "pages": pages,
    }

    content = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    stats = {
        "page_count": page_count,
        "non_empty_pages": non_empty_pages,
        "text_len": total_text_len,
    }
    return content, stats

"""
Preflight de PDFs antes de OCR.

Clasifica el documento en cuatro categorías operativas:
  - unsigned_text
  - signed_text
  - signed_no_text
  - unsigned_no_text

La política del worker usa esa clasificación para decidir si:
  - procesa OCR (`process`)
  - omite OCR porque el PDF ya es útil (`skip`)
  - bloquea el job por firma digital (`block`)
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import pikepdf

_WORD_RE = re.compile(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]{2,}\b", re.UNICODE)
_ALNUM_RE = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", re.UNICODE)


def inspect_pdf(pdf_path: Path, *, max_pages: int = 8) -> dict:
    """
    Ejecuta preflight conservador sobre un PDF local.

    Heurística de texto útil:
      - analiza hasta `max_pages`
      - considera útil una página con >=120 caracteres alfanuméricos
        y >=20 palabras
      - considera útil el documento si:
        * el total analizado tiene >=400 caracteres y >=80 palabras, o
        * una sola página es útil en un documento de una página, o
        * dos o más páginas son útiles
    """
    signed, signature_indicators = _detect_digital_signature(pdf_path)
    text_info = _detect_useful_text(pdf_path, max_pages=max_pages)
    has_useful_text = text_info["has_useful_text"]

    if signed and has_useful_text:
        classification = "signed_text"
        decision = "skip"
        reason_code = "signed_pdf_with_text"
        message = "PDF firmado digitalmente y con texto util: no requiere OCR"
    elif signed and not has_useful_text:
        classification = "signed_no_text"
        decision = "block"
        reason_code = "digital_signature_blocked"
        message = "PDF bloqueado por firma digital y sin texto util"
    elif not signed and has_useful_text:
        classification = "unsigned_text"
        decision = "skip"
        reason_code = "useful_text_present"
        message = "PDF con texto util: no requiere OCR"
    else:
        classification = "unsigned_no_text"
        decision = "process"
        reason_code = None
        message = None

    return {
        "classification": classification,
        "decision": decision,
        "reason_code": reason_code,
        "message": message,
        "signed": signed,
        "has_useful_text": has_useful_text,
        "signature_indicators": signature_indicators,
        "text_analysis": text_info,
    }


def _detect_useful_text(pdf_path: Path, *, max_pages: int = 8) -> dict:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    analyzed_pages = min(len(doc), max_pages)
    page_stats: list[dict] = []
    useful_pages = 0
    total_chars = 0
    total_words = 0

    try:
        for index in range(analyzed_pages):
            page = doc[index]
            text = (page.get_text("text") or "").strip()
            chars = len(_ALNUM_RE.findall(text))
            words = len(_WORD_RE.findall(text))
            page_useful = chars >= 120 and words >= 20

            if page_useful:
                useful_pages += 1
            total_chars += chars
            total_words += words
            page_stats.append({
                "page": index + 1,
                "chars": chars,
                "words": words,
                "useful": page_useful,
            })
    finally:
        doc.close()

    has_useful_text = (
        (analyzed_pages <= 1 and useful_pages >= 1)
        or useful_pages >= 2
        or (total_chars >= 400 and total_words >= 80)
    )

    return {
        "analyzed_pages": analyzed_pages,
        "total_pages": total_pages,
        "total_chars": total_chars,
        "total_words": total_words,
        "useful_pages": useful_pages,
        "has_useful_text": has_useful_text,
        "pages": page_stats,
    }


def _detect_digital_signature(pdf_path: Path) -> tuple[bool, list[str]]:
    indicators: list[str] = []

    try:
        with pikepdf.open(pdf_path) as pdf:
            root = pdf.Root

            perms = root.get("/Perms")
            if perms:
                for key in ("/DocMDP", "/UR", "/UR3"):
                    if perms.get(key) is not None:
                        indicators.append(f"Perms:{key}")

            acro_form = root.get("/AcroForm")
            if acro_form:
                fields = list(acro_form.get("/Fields", []))
                if _fields_have_signature(fields):
                    indicators.append("AcroForm:/Sig")
    except Exception as exc:
        indicators.append(f"signature_detection_error:{exc.__class__.__name__}")

    return bool([item for item in indicators if not item.startswith("signature_detection_error:")]), indicators


def _fields_have_signature(fields: list) -> bool:
    stack = list(fields)
    visited: set[int] = set()

    while stack:
        field = stack.pop()
        try:
            obj = field.get_object() if hasattr(field, "get_object") else field
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            field_type = obj.get("/FT")
            if str(field_type) == "/Sig":
                return True

            value = obj.get("/V")
            if value is not None:
                value_obj = value.get_object() if hasattr(value, "get_object") else value
                if str(value_obj.get("/Type")) == "/Sig":
                    return True

            kids = obj.get("/Kids")
            if kids:
                stack.extend(list(kids))
        except Exception:
            continue

    return False

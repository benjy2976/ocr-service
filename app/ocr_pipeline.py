import os
import time
import uuid
import subprocess
from pathlib import Path

import requests
import fitz

DEFAULT_TMP_DIR = os.getenv("OCR_TMP_DIR", "/data/tmp")
DEFAULT_OUT_DIR = os.getenv("OCR_OUT_DIR", "/data/out")
DEFAULT_LANG = os.getenv("OCR_LANG", "spa")
DEFAULT_MODE = os.getenv("OCR_MODE", "searchable_cpu")


def _safe_filename_from_url(url: str) -> str:
    name = url.split("/")[-1] or "file.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def _download(url: str, dst: Path) -> tuple[str | None, int]:
    size = 0
    content_type = None
    headers = {
        "User-Agent": "Mozilla/5.0 (OCR-Pilot)",
        "Accept": "application/pdf,*/*",
    }
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type")
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    size += len(chunk)
                    f.write(chunk)
    return content_type, size


def _copy_local(src: Path, dst: Path) -> tuple[str | None, int]:
    size = 0
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            chunk = fsrc.read(1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)
            size += len(chunk)
    return "application/pdf", size


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


def _extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts).strip()


def _ocr_searchable_cpu(input_pdf: Path, output_pdf: Path, lang: str) -> None:
    # OCRmyPDF uses Tesseract under the hood (CPU). We keep optimize=0 to avoid recompression.
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize", "0",
        "-l", lang,
        str(input_pdf),
        str(output_pdf),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(f"ocrmypdf failed. stdout={stdout} stderr={stderr}")


def run_ocr(url: str, mode: str | None = None, lang: str | None = None) -> dict:
    mode = mode or DEFAULT_MODE
    lang = lang or DEFAULT_LANG

    tmp_dir = Path(DEFAULT_TMP_DIR)
    out_dir = Path(DEFAULT_OUT_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:12]
    src_name = _safe_filename_from_url(url)
    src_pdf = tmp_dir / f"{token}_{src_name}"

    content_type, size = _download(url, src_pdf)

    if not _is_pdf(src_pdf):
        raise RuntimeError(
            f"Downloaded content is not a PDF. content_type={content_type} size={size} url={url}"
        )

    start = time.time()

    if mode == "searchable_cpu":
        out_pdf = out_dir / f"{token}_searchable.pdf"
        _ocr_searchable_cpu(src_pdf, out_pdf, lang)
        text = _extract_text(out_pdf)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    elapsed = time.time() - start

    return {
        "mode": mode,
        "lang": lang,
        "source": str(src_pdf),
        "output_pdf": str(out_pdf),
        "content_type": content_type,
        "source_bytes": size,
        "text_len": len(text),
        "elapsed_sec": round(elapsed, 2),
    }


def run_ocr_file(input_pdf: Path, mode: str | None = None, lang: str | None = None) -> dict:
    mode = mode or DEFAULT_MODE
    lang = lang or DEFAULT_LANG

    tmp_dir = Path(DEFAULT_TMP_DIR)
    out_dir = Path(DEFAULT_OUT_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:12]
    src_pdf = tmp_dir / f"{token}_{input_pdf.name}"

    content_type, size = _copy_local(input_pdf, src_pdf)

    if not _is_pdf(src_pdf):
        raise RuntimeError(
            f"Input file is not a PDF. content_type={content_type} size={size} path={input_pdf}"
        )

    start = time.time()

    if mode == "searchable_cpu":
        out_pdf = out_dir / f"{token}_searchable.pdf"
        _ocr_searchable_cpu(src_pdf, out_pdf, lang)
        text = _extract_text(out_pdf)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    elapsed = time.time() - start

    return {
        "mode": mode,
        "lang": lang,
        "source": str(src_pdf),
        "output_pdf": str(out_pdf),
        "content_type": content_type,
        "source_bytes": size,
        "text_len": len(text),
        "elapsed_sec": round(elapsed, 2),
    }

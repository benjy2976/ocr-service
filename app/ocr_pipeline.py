import os
import time
import uuid
import subprocess
import tempfile
from pathlib import Path

import requests
import fitz
import numpy as np
import cv2

DEFAULT_TMP_DIR = os.getenv("OCR_TMP_DIR", "/data/tmp")
DEFAULT_OUT_DIR = os.getenv("OCR_OUT_DIR", "/data/out")
DEFAULT_LANG = os.getenv("OCR_LANG", "spa")
DEFAULT_MODE = os.getenv("OCR_MODE", "searchable_cpu")
DEFAULT_DESKEW = os.getenv("OCR_DESKEW", "false").lower() in ("1", "true", "yes")
DEFAULT_CLEAN = os.getenv("OCR_CLEAN", "false").lower() in ("1", "true", "yes")
DEFAULT_REMOVE_VECTORS = os.getenv("OCR_REMOVE_VECTORS", "false").lower() in ("1", "true", "yes")
DEFAULT_PSM = os.getenv("OCR_TESSERACT_PSM")
DEFAULT_MASK_STAMPS = os.getenv("OCR_MASK_STAMPS", "false").lower() in ("1", "true", "yes")
DEFAULT_MASK_SIGNATURES = os.getenv("OCR_MASK_SIGNATURES", "false").lower() in ("1", "true", "yes")
DEFAULT_STAMP_MIN_AREA = float(os.getenv("OCR_STAMP_MIN_AREA", "20000"))
DEFAULT_STAMP_MAX_AREA = float(os.getenv("OCR_STAMP_MAX_AREA", "400000"))
DEFAULT_STAMP_CIRCULARITY = float(os.getenv("OCR_STAMP_CIRCULARITY", "0.5"))
DEFAULT_STAMP_RECT_ASPECT_MIN = float(os.getenv("OCR_STAMP_RECT_ASPECT_MIN", "0.5"))
DEFAULT_STAMP_RECT_ASPECT_MAX = float(os.getenv("OCR_STAMP_RECT_ASPECT_MAX", "2.0"))
DEFAULT_SIGNATURE_REGION = float(os.getenv("OCR_SIGNATURE_REGION", "0.35"))


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


def _mask_stamps_and_signatures(
    image_bgr: np.ndarray,
    *,
    mask_stamps: bool,
    mask_signatures: bool,
    stamp_min_area: float,
    stamp_max_area: float,
    stamp_circularity: float,
    stamp_rect_aspect_min: float,
    stamp_rect_aspect_max: float,
    signature_region: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if mask_stamps:
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < stamp_min_area or area > stamp_max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * (area / (peri * peri))
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0.0
            is_circle = circularity >= stamp_circularity
            is_rect = stamp_rect_aspect_min <= aspect <= stamp_rect_aspect_max
            if is_circle or is_rect:
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    if mask_signatures:
        start_y = int(h * (1.0 - signature_region))
        roi = bin_img[start_y:h, :]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.2 and bh < h * 0.08:
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED, offset=(0, start_y))

    if mask.sum() == 0:
        return image_bgr

    masked = image_bgr.copy()
    masked[mask == 255] = (255, 255, 255)
    return masked


def _prepare_masked_pdf(
    input_pdf: Path,
    *,
    mask_stamps: bool,
    mask_signatures: bool,
    stamp_min_area: float,
    stamp_max_area: float,
    stamp_circularity: float,
    stamp_rect_aspect_min: float,
    stamp_rect_aspect_max: float,
    signature_region: float,
) -> Path:
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()

    for page in doc:
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = _mask_stamps_and_signatures(
            img,
            mask_stamps=mask_stamps,
            mask_signatures=mask_signatures,
            stamp_min_area=stamp_min_area,
            stamp_max_area=stamp_max_area,
            stamp_circularity=stamp_circularity,
            stamp_rect_aspect_min=stamp_rect_aspect_min,
            stamp_rect_aspect_max=stamp_rect_aspect_max,
            signature_region=signature_region,
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(rect, stream=img_bytes)

    tmp_dir = Path(DEFAULT_TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="masked_", suffix=".pdf", dir=str(tmp_dir))
    os.close(fd)
    out_path = Path(path)
    out_doc.save(out_path)
    out_doc.close()
    doc.close()
    return out_path


def _ocr_searchable_cpu(
    input_pdf: Path,
    output_pdf: Path,
    lang: str,
    *,
    deskew: bool,
    clean: bool,
    remove_vectors: bool,
    psm: str | None,
) -> None:
    # OCRmyPDF uses Tesseract under the hood (CPU). We keep optimize=0 to avoid recompression.
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize", "0",
        "-l", lang,
    ]
    if deskew:
        cmd.append("--deskew")
    if clean:
        cmd.append("--clean")
    if remove_vectors:
        cmd.append("--remove-vectors")
    if psm:
        cmd.extend(["--tesseract-pagesegmode", str(psm)])
    cmd.extend([
        str(input_pdf),
        str(output_pdf),
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(f"ocrmypdf failed. stdout={stdout} stderr={stderr}")


def run_ocr(
    url: str,
    mode: str | None = None,
    lang: str | None = None,
    *,
    deskew: bool | None = None,
    clean: bool | None = None,
    remove_vectors: bool | None = None,
    psm: str | None = None,
    mask_stamps: bool | None = None,
    mask_signatures: bool | None = None,
    stamp_min_area: float | None = None,
    stamp_max_area: float | None = None,
    stamp_circularity: float | None = None,
    stamp_rect_aspect_min: float | None = None,
    stamp_rect_aspect_max: float | None = None,
    signature_region: float | None = None,
) -> dict:
    mode = mode or DEFAULT_MODE
    lang = lang or DEFAULT_LANG
    deskew = DEFAULT_DESKEW if deskew is None else deskew
    clean = DEFAULT_CLEAN if clean is None else clean
    remove_vectors = DEFAULT_REMOVE_VECTORS if remove_vectors is None else remove_vectors
    psm = DEFAULT_PSM if psm is None else psm
    mask_stamps = DEFAULT_MASK_STAMPS if mask_stamps is None else mask_stamps
    mask_signatures = DEFAULT_MASK_SIGNATURES if mask_signatures is None else mask_signatures
    stamp_min_area = DEFAULT_STAMP_MIN_AREA if stamp_min_area is None else stamp_min_area
    stamp_max_area = DEFAULT_STAMP_MAX_AREA if stamp_max_area is None else stamp_max_area
    stamp_circularity = (
        DEFAULT_STAMP_CIRCULARITY if stamp_circularity is None else stamp_circularity
    )
    stamp_rect_aspect_min = (
        DEFAULT_STAMP_RECT_ASPECT_MIN
        if stamp_rect_aspect_min is None
        else stamp_rect_aspect_min
    )
    stamp_rect_aspect_max = (
        DEFAULT_STAMP_RECT_ASPECT_MAX
        if stamp_rect_aspect_max is None
        else stamp_rect_aspect_max
    )
    signature_region = (
        DEFAULT_SIGNATURE_REGION if signature_region is None else signature_region
    )

    out_dir = Path(DEFAULT_OUT_DIR)
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
        ocr_input = src_pdf
        if mask_stamps or mask_signatures:
            ocr_input = _prepare_masked_pdf(
                src_pdf,
                mask_stamps=mask_stamps,
                mask_signatures=mask_signatures,
                stamp_min_area=stamp_min_area,
                stamp_max_area=stamp_max_area,
                stamp_circularity=stamp_circularity,
                stamp_rect_aspect_min=stamp_rect_aspect_min,
                stamp_rect_aspect_max=stamp_rect_aspect_max,
                signature_region=signature_region,
            )
        _ocr_searchable_cpu(
            ocr_input,
            out_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
        )
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


def run_ocr_file(
    input_pdf: Path,
    mode: str | None = None,
    lang: str | None = None,
    *,
    deskew: bool | None = None,
    clean: bool | None = None,
    remove_vectors: bool | None = None,
    psm: str | None = None,
    mask_stamps: bool | None = None,
    mask_signatures: bool | None = None,
    stamp_min_area: float | None = None,
    stamp_max_area: float | None = None,
    stamp_circularity: float | None = None,
    stamp_rect_aspect_min: float | None = None,
    stamp_rect_aspect_max: float | None = None,
    signature_region: float | None = None,
) -> dict:
    mode = mode or DEFAULT_MODE
    lang = lang or DEFAULT_LANG
    deskew = DEFAULT_DESKEW if deskew is None else deskew
    clean = DEFAULT_CLEAN if clean is None else clean
    remove_vectors = DEFAULT_REMOVE_VECTORS if remove_vectors is None else remove_vectors
    psm = DEFAULT_PSM if psm is None else psm
    mask_stamps = DEFAULT_MASK_STAMPS if mask_stamps is None else mask_stamps
    mask_signatures = DEFAULT_MASK_SIGNATURES if mask_signatures is None else mask_signatures
    stamp_min_area = DEFAULT_STAMP_MIN_AREA if stamp_min_area is None else stamp_min_area
    stamp_max_area = DEFAULT_STAMP_MAX_AREA if stamp_max_area is None else stamp_max_area
    stamp_circularity = (
        DEFAULT_STAMP_CIRCULARITY if stamp_circularity is None else stamp_circularity
    )
    stamp_rect_aspect_min = (
        DEFAULT_STAMP_RECT_ASPECT_MIN
        if stamp_rect_aspect_min is None
        else stamp_rect_aspect_min
    )
    stamp_rect_aspect_max = (
        DEFAULT_STAMP_RECT_ASPECT_MAX
        if stamp_rect_aspect_max is None
        else stamp_rect_aspect_max
    )
    signature_region = (
        DEFAULT_SIGNATURE_REGION if signature_region is None else signature_region
    )

    tmp_dir = Path(DEFAULT_TMP_DIR)
    out_dir = Path(DEFAULT_OUT_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:12]
    src_pdf = input_pdf
    content_type = "application/pdf"
    size = src_pdf.stat().st_size

    if not _is_pdf(src_pdf):
        raise RuntimeError(
            f"Input file is not a PDF. content_type={content_type} size={size} path={input_pdf}"
        )

    start = time.time()

    if mode == "searchable_cpu":
        out_pdf = out_dir / f"{token}_searchable.pdf"
        ocr_input = src_pdf
        if mask_stamps or mask_signatures:
            ocr_input = _prepare_masked_pdf(
                src_pdf,
                mask_stamps=mask_stamps,
                mask_signatures=mask_signatures,
                stamp_min_area=stamp_min_area,
                stamp_max_area=stamp_max_area,
                stamp_circularity=stamp_circularity,
                stamp_rect_aspect_min=stamp_rect_aspect_min,
                stamp_rect_aspect_max=stamp_rect_aspect_max,
                signature_region=signature_region,
            )
        _ocr_searchable_cpu(
            ocr_input,
            out_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
        )
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

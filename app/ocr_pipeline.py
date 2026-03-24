import os
import json
import time
import uuid
import subprocess
import tempfile
import hashlib
import re
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
DEFAULT_MASK_DILATE = int(os.getenv("OCR_MASK_DILATE", "4"))
DEFAULT_MASK_GRAYSCALE = os.getenv("OCR_MASK_GRAYSCALE", "false").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_MASK_USE_DETECTOR = os.getenv("OCR_MASK_USE_DETECTOR", "auto").lower()
DEFAULT_MASK_DETECTOR_CONF = float(os.getenv("OCR_MASK_DETECTOR_CONF", "0.25"))
DEFAULT_MASK_DETECTOR_IMGSZ = int(os.getenv("OCR_MASK_DETECTOR_IMGSZ", "1024"))
DEFAULT_MASK_DETECTOR_CLASSES = [
    c.strip()
    for c in os.getenv("OCR_MASK_DETECTOR_CLASSES", "").split(",")
    if c.strip()
]
DEFAULT_MASK_COLOR_SAT_MIN = int(os.getenv("OCR_MASK_COLOR_SAT_MIN", "40"))
DEFAULT_MASK_COLOR_VAL_MIN = int(os.getenv("OCR_MASK_COLOR_VAL_MIN", "40"))
DEFAULT_MASK_TEXT_THR = int(os.getenv("OCR_MASK_TEXT_THR", "80"))
DEFAULT_MASK_TEXT_DILATE = int(os.getenv("OCR_MASK_TEXT_DILATE", "2"))
DEFAULT_MASK_MORPH = int(os.getenv("OCR_MASK_MORPH", "3"))
DEFAULT_MASK_STRATEGY = os.getenv("OCR_MASK_STRATEGY", "hybrid").lower()
DEFAULT_MASK_TEXT_DENSITY = float(os.getenv("OCR_MASK_TEXT_DENSITY", "0.015"))
DEFAULT_MASK_BLUR_K = int(os.getenv("OCR_MASK_BLUR_K", "19"))
DEFAULT_MASK_INPAINT_RADIUS = int(os.getenv("OCR_MASK_INPAINT_RADIUS", "3"))
DEFAULT_MASK_INPAINT_METHOD = os.getenv("OCR_MASK_INPAINT_METHOD", "telea").lower()
DEFAULT_MASK_DPI = int(os.getenv("OCR_MASK_DPI", "300"))
DEFAULT_MASK_CLASS_PARAMS = {}
_MASK_CLASS_PARAMS_PATH = os.getenv("OCR_MASK_CLASS_PARAMS_PATH", "").strip()
_MASK_CLASS_PARAMS_RAW = os.getenv("OCR_MASK_CLASS_PARAMS", "").strip()
if _MASK_CLASS_PARAMS_PATH:
    try:
        with open(_MASK_CLASS_PARAMS_PATH, "r", encoding="utf-8") as f:
            DEFAULT_MASK_CLASS_PARAMS = json.load(f)
    except Exception:
        DEFAULT_MASK_CLASS_PARAMS = {}
elif _MASK_CLASS_PARAMS_RAW:
    try:
        DEFAULT_MASK_CLASS_PARAMS = json.loads(_MASK_CLASS_PARAMS_RAW)
    except Exception:
        DEFAULT_MASK_CLASS_PARAMS = {}
DEFAULT_MASK_SAVE_DETECTIONS = os.getenv("OCR_MASK_SAVE_DETECTIONS", "true").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_STAMP_MODEL_PATH = os.getenv(
    "STAMP_MODEL_PATH",
    "/data/models/stamp_detector.pt",
)
_STAMP_MODEL_V2_PATH = Path("/data/models/stamp_detector_v2.pt")
if (
    DEFAULT_STAMP_MODEL_PATH == "/data/models/stamp_detector.pt"
    and _STAMP_MODEL_V2_PATH.exists()
):
    DEFAULT_STAMP_MODEL_PATH = str(_STAMP_MODEL_V2_PATH)
DEFAULT_STAMP_TEST_PDF = os.getenv(
    "STAMP_TEST_PDF",
    "/data/samples/100/00000010000432026_1771454832.pdf",
)
DEFAULT_STAMP_TEST_DPI = int(os.getenv("STAMP_TEST_DPI", "200"))
DEFAULT_STAMP_TEST_CONF = float(os.getenv("STAMP_TEST_CONF", "0.25"))
DEFAULT_STAMP_TEST_IMGSZ = int(os.getenv("STAMP_TEST_IMGSZ", "640"))
DEFAULT_STAMP_DEVICE = os.getenv("STAMP_DEVICE", "auto")
DEFAULT_PADDLE_DPI = int(os.getenv("OCR_PADDLE_DPI", "300"))
DEFAULT_PADDLE_LANG = os.getenv("OCR_PADDLE_LANG", "es")
_OCR_JOBS_RAW = os.getenv("OCR_JOBS", "").strip()
DEFAULT_OCR_JOBS = None
if _OCR_JOBS_RAW:
    try:
        DEFAULT_OCR_JOBS = max(1, int(_OCR_JOBS_RAW))
    except ValueError:
        DEFAULT_OCR_JOBS = None

_STAMP_DETECTOR = None
_STAMP_DETECTOR_LABELS = None
_TEXT_BLOCK_DETECTOR = None
_TEXT_BLOCK_DETECTOR_PATH = None


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return slug.strip("._") or "default"


def _text_pages_dir() -> Path:
    return Path(DEFAULT_OUT_DIR) / "text_pages"


def _text_block_model_path() -> Path | None:
    raw = os.getenv("OCR_TEXT_BLOCK_MODEL_PATH", "").strip() or os.getenv("TEXT_REVIEW_MODEL_PATH", "").strip()
    if raw:
        path = Path(raw)
        return path if path.exists() else None
    models_dir = _text_pages_dir() / "models"
    if not models_dir.exists():
        return None
    candidates = [p for p in models_dir.glob("*.pt") if p.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _load_text_block_detector():
    global _TEXT_BLOCK_DETECTOR, _TEXT_BLOCK_DETECTOR_PATH
    model_path = _text_block_model_path()
    if model_path is None:
        return None, None
    model_path = model_path.resolve()
    if _TEXT_BLOCK_DETECTOR is not None and _TEXT_BLOCK_DETECTOR_PATH == model_path:
        return _TEXT_BLOCK_DETECTOR, model_path
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is not installed in the service") from exc
    _TEXT_BLOCK_DETECTOR = YOLO(str(model_path))
    _TEXT_BLOCK_DETECTOR_PATH = model_path
    return _TEXT_BLOCK_DETECTOR, model_path


def _resolve_text_block_device() -> str:
    raw = os.getenv("OCR_TEXT_BLOCK_DEVICE", "").strip().lower() or os.getenv("TEXT_REVIEW_MODEL_DEVICE", "auto").strip().lower()
    if raw and raw != "auto":
        return raw
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_TEXT_BLOCK_CONF = float(
    os.getenv("OCR_TEXT_BLOCK_CONF", "").strip() or os.getenv("TEXT_REVIEW_MODEL_CONF", "0.25")
)
DEFAULT_TEXT_BLOCK_IMGSZ = int(
    os.getenv("OCR_TEXT_BLOCK_IMGSZ", "").strip() or os.getenv("TEXT_REVIEW_MODEL_IMGSZ", "960")
)


def _load_stamp_detector():
    global _STAMP_DETECTOR, _STAMP_DETECTOR_LABELS
    if _STAMP_DETECTOR is not None:
        return _STAMP_DETECTOR, _STAMP_DETECTOR_LABELS
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ultralytics is not installed in the service") from exc

    model_path = Path(DEFAULT_STAMP_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"STAMP_MODEL_PATH not found: {model_path}")
    model = YOLO(str(model_path))
    _STAMP_DETECTOR = model
    _STAMP_DETECTOR_LABELS = model.names or {}
    return _STAMP_DETECTOR, _STAMP_DETECTOR_LABELS


def _resolve_detector_device() -> str:
    device = DEFAULT_STAMP_DEVICE
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    return device


def _detect_stamp_boxes(
    image_bgr: np.ndarray,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
) -> list[dict]:
    model, labels = _load_stamp_detector()
    device = _resolve_detector_device()
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source=img_rgb,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    if not results:
        return []
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls.item()) if hasattr(b, "cls") else None
            label = labels.get(cls_id, str(cls_id)) if cls_id is not None else ""
            if allowed_classes and label not in allowed_classes:
                continue
            conf_score = float(b.conf.item()) if hasattr(b, "conf") else None
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "conf": conf_score,
                }
            )
    return boxes


def _detect_text_block_boxes(
    image_bgr: np.ndarray,
    conf: float,
    imgsz: int,
) -> tuple[list[dict], str | None]:
    model, model_path = _load_text_block_detector()
    if model is None or model_path is None:
        return [], None
    device = _resolve_text_block_device()
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    boxes: list[dict] = []
    if results:
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    boxes.sort(key=lambda b: (round(b["y1"], 3), round(b["x1"], 3)))
    return boxes, model_path.stem


def _color_mask_hsv(
    roi_bgr: np.ndarray,
    sat_min: int,
    val_min: int,
    morph: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, sat_min, val_min), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, sat_min, val_min), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, sat_min, val_min), (135, 255, 255))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.bitwise_or(mask, blue)
    if morph and morph > 1:
        k = int(morph)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _text_protection_mask(gray: np.ndarray, thr: int, dilate: int) -> np.ndarray:
    text = (gray < thr).astype(np.uint8) * 255
    if dilate and dilate > 0:
        k = int(dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        text = cv2.dilate(text, kernel, iterations=1)
    return text


def _text_density(gray: np.ndarray, thr: int) -> float:
    if gray.size == 0:
        return 0.0
    text = gray < thr
    return float(text.sum()) / float(text.size)


def _intensity_mask(gray: np.ndarray, morph: int) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if morph and morph > 1:
        k = int(morph)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _apply_blur_mask(roi: np.ndarray, mask: np.ndarray, blur_k: int) -> np.ndarray:
    k = max(3, int(blur_k))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    out = roi.copy()
    out[mask == 255] = blurred[mask == 255]
    return out


def _apply_inpaint(roi: np.ndarray, mask: np.ndarray, radius: int, method: str) -> np.ndarray:
    if mask.sum() == 0:
        return roi
    r = max(1, int(radius))
    m = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(roi, mask, r, m)


def _apply_detector_mask(
    image_bgr: np.ndarray,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
    sat_min: int,
    val_min: int,
    text_thr: int,
    text_dilate: int,
    morph: int,
    mask_dilate: int,
    strategy: str,
    text_density_thr: float,
    blur_k: int,
    inpaint_radius: int,
    inpaint_method: str,
) -> tuple[np.ndarray | None, list[dict]]:
    h, w = image_bgr.shape[:2]
    boxes = _detect_stamp_boxes(image_bgr, conf=conf, imgsz=imgsz, allowed_classes=allowed_classes)
    if not boxes:
        return None, []
    out = image_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = box.get("label") or ""
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # Per-class overrides
        cls_params = DEFAULT_MASK_CLASS_PARAMS.get(label, {}) if DEFAULT_MASK_CLASS_PARAMS else {}
        cls_strategy = cls_params.get("strategy", strategy)
        cls_text_density = float(cls_params.get("text_density", text_density_thr))
        cls_blur_k = int(cls_params.get("blur_k", blur_k))
        cls_inpaint_radius = int(cls_params.get("inpaint_radius", inpaint_radius))
        cls_inpaint_method = str(cls_params.get("inpaint_method", inpaint_method)).lower()
        cls_sat_min = int(cls_params.get("sat_min", sat_min))
        cls_val_min = int(cls_params.get("val_min", val_min))
        cls_text_thr = int(cls_params.get("text_thr", text_thr))
        cls_text_dilate = int(cls_params.get("text_dilate", text_dilate))
        cls_morph = int(cls_params.get("morph", morph))
        cls_mask_dilate = int(cls_params.get("mask_dilate", mask_dilate))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        color_mask = _color_mask_hsv(roi, sat_min=cls_sat_min, val_min=cls_val_min, morph=cls_morph)
        if color_mask.sum() == 0 or label == "logo":
            color_mask = _intensity_mask(gray, morph=cls_morph)
        if cls_mask_dilate and color_mask.sum() > 0:
            k = max(1, int(cls_mask_dilate))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            color_mask = cv2.dilate(color_mask, kernel, iterations=1)

        text_density = _text_density(gray, cls_text_thr)
        if cls_strategy == "none":
            continue
        use_inpaint = cls_strategy == "inpaint" or (
            cls_strategy == "hybrid" and text_density >= cls_text_density
        )
        use_blur = cls_strategy == "blur" or (
            cls_strategy == "hybrid" and not use_inpaint
        )

        if use_inpaint:
            # In inpaint mode we avoid subtracting text to let reconstruction happen.
            roi_out = _apply_inpaint(
                roi,
                color_mask,
                radius=cls_inpaint_radius,
                method=cls_inpaint_method,
            )
        elif use_blur:
            text_mask = _text_protection_mask(
                gray,
                thr=cls_text_thr,
                dilate=cls_text_dilate,
            )
            mask = color_mask.copy()
            mask[text_mask == 255] = 0
            roi_out = _apply_blur_mask(roi, mask, blur_k=cls_blur_k)
        else:
            roi_out = roi

        out[y1:y2, x1:x2] = roi_out

    return out, boxes


def _render_detections_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
) -> Path:
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    for page in doc:
        mat = fitz.Matrix(DEFAULT_MASK_DPI / 72, DEFAULT_MASK_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        boxes = _detect_stamp_boxes(img, conf=conf, imgsz=imgsz, allowed_classes=allowed_classes)
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = box.get("label") or ""
            score = box.get("conf")
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if score is not None:
                text = f"{label} {score:.2f}"
            else:
                text = label
            if text:
                cv2.putText(
                    img,
                    text,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(rect, stream=img_bytes)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(output_pdf)
    out_doc.close()
    doc.close()
    return output_pdf




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
    mask_dilate: int,
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

    if mask_dilate and mask.sum() > 0:
        k = max(1, int(mask_dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if mask.sum() == 0:
        return image_bgr

    masked = image_bgr.copy()
    masked[mask == 255] = (255, 255, 255)
    return masked


def _detect_text_line_boxes(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_w = max(25, w // 35)
    kernel_h = max(3, h // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    min_w = max(80, int(w * 0.08))
    min_h = max(10, int(h * 0.006))
    max_h = max(40, int(h * 0.06))
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_w or bh < min_h or bh > max_h:
            continue
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _intersection_ratio(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    area = float(min(aw * ah, bw * bh))
    if area <= 0:
        return 0.0
    return inter / area


def _stamp_color_mask(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    color_mask = _color_mask_hsv(
        roi_bgr,
        sat_min=DEFAULT_MASK_COLOR_SAT_MIN,
        val_min=DEFAULT_MASK_COLOR_VAL_MIN,
        morph=max(1, DEFAULT_MASK_MORPH),
    )
    if color_mask.sum() == 0:
        color_mask = _intensity_mask(gray, morph=max(1, DEFAULT_MASK_MORPH))
    return color_mask


def _protect_document_text_mask(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = _text_protection_mask(
        gray,
        thr=DEFAULT_MASK_TEXT_THR,
        dilate=max(1, DEFAULT_MASK_TEXT_DILATE),
    )

    h, w = gray.shape[:2]
    kernel_w = max(15, w // 6)
    kernel_h = max(1, h // 20)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    line_like = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, line_kernel, iterations=1)
    return cv2.bitwise_or(dark_mask, line_like)


def _apply_conservative_mask_to_box(
    roi_bgr: np.ndarray,
    *,
    intersects_text: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stamp_mask = _stamp_color_mask(roi_bgr)
    if stamp_mask.sum() == 0:
        return roi_bgr, stamp_mask

    if intersects_text:
        protected = _protect_document_text_mask(roi_bgr)
        mask = stamp_mask.copy()
        mask[protected == 255] = 0
        if mask.sum() == 0:
            return roi_bgr, mask
        out = roi_bgr.copy()
        out[mask == 255] = (255, 255, 255)
        return out, mask

    out = _apply_inpaint(
        roi_bgr,
        stamp_mask,
        radius=max(1, DEFAULT_MASK_INPAINT_RADIUS),
        method=DEFAULT_MASK_INPAINT_METHOD,
    )
    return out, stamp_mask


def _prepare_conservative_working_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
    grayscale: bool,
    detected_pdf: Path | None = None,
) -> Path:
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    detected_doc = fitz.open() if detected_pdf else None

    for page in doc:
        mat = fitz.Matrix(DEFAULT_MASK_DPI / 72, DEFAULT_MASK_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_color = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        img = img_color.copy()
        text_boxes, _text_model_name = _detect_text_block_boxes(
            img_color,
            conf=DEFAULT_TEXT_BLOCK_CONF,
            imgsz=DEFAULT_TEXT_BLOCK_IMGSZ,
        )
        text_rects = [
            fitz.Rect(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
            for box in text_boxes
        ]
        if not text_rects:
            text_lines = _detect_text_line_boxes(img_color)
            text_rects = [
                fitz.Rect(float(x), float(y), float(x + w), float(y + h))
                for x, y, w, h in text_lines
            ]
        boxes = _detect_stamp_boxes(
            img_color,
            conf=conf,
            imgsz=imgsz,
            allowed_classes=allowed_classes,
        )

        applied_boxes = []
        page_rect = fitz.Rect(0, 0, img.shape[1], img.shape[0])
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            x1 = max(0, min(img.shape[1] - 1, x1))
            y1 = max(0, min(img.shape[0] - 1, y1))
            x2 = max(0, min(img.shape[1], x2))
            y2 = max(0, min(img.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                continue

            det_rect = fitz.Rect(float(x1), float(y1), float(x2), float(y2))
            redactions, segments = _build_ocr_redaction_rects(
                page_rect,
                det_rect,
                text_rects=text_rects,
                shrink_factor=0.78,
                detector_label=box.get("label") or "",
            )
            masked_pixels = 0
            for red in redactions:
                rx0 = max(0, min(img.shape[1], int(np.floor(red.x0))))
                ry0 = max(0, min(img.shape[0], int(np.floor(red.y0))))
                rx1 = max(0, min(img.shape[1], int(np.ceil(red.x1))))
                ry1 = max(0, min(img.shape[0], int(np.ceil(red.y1))))
                if rx1 <= rx0 or ry1 <= ry0:
                    continue
                masked_pixels += max(0, rx1 - rx0) * max(0, ry1 - ry0)
                img[ry0:ry1, rx0:rx1] = (255, 255, 255)
            applied_boxes.append(
                {
                    **box,
                    "mask_pixels": int(masked_pixels),
                    "redactions": [fitz.Rect(r) for r in redactions],
                    "segments": segments,
                }
            )

        if detected_doc is not None:
            vis = img_color.copy()
            for box in applied_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                color = (0, 255, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = box.get("label") or ""
                score = box.get("conf")
                text = f"{label} {score:.2f}" if score is not None else label
                cv2.putText(
                    vis,
                    text,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                for red in box.get("redactions") or []:
                    cv2.rectangle(
                        vis,
                        (int(round(red.x0)), int(round(red.y0))),
                        (int(round(red.x1)), int(round(red.y1))),
                        (0, 165, 255),
                        2,
                    )
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vis_bytes = cv2.imencode(".png", vis_rgb)[1].tobytes()
            rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
            det_page = detected_doc.new_page(width=page.rect.width, height=page.rect.height)
            det_page.insert_image(rect, stream=vis_bytes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(rect, stream=img_bytes)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(output_pdf)
    out_doc.close()
    if detected_doc is not None and detected_pdf is not None:
        detected_pdf.parent.mkdir(parents=True, exist_ok=True)
        detected_doc.save(detected_pdf)
        detected_doc.close()
    doc.close()
    return output_pdf


def _prepare_masked_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    mask_stamps: bool,
    mask_signatures: bool,
    stamp_min_area: float,
    stamp_max_area: float,
    stamp_circularity: float,
    stamp_rect_aspect_min: float,
    stamp_rect_aspect_max: float,
    signature_region: float,
    grayscale: bool,
    mask_dilate: int,
    detected_pdf: Path | None = None,
) -> Path:
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    detected_doc = fitz.open() if detected_pdf else None

    for page in doc:
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_color = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        img = img_color
        use_detector = DEFAULT_MASK_USE_DETECTOR
        if use_detector == "auto":
            use_detector = "true" if Path(DEFAULT_STAMP_MODEL_PATH).exists() else "false"
        used_detector = False
        det_boxes = []
        if use_detector in ("1", "true", "yes"):
            try:
                allowed = set(DEFAULT_MASK_DETECTOR_CLASSES) if DEFAULT_MASK_DETECTOR_CLASSES else None
                det_img, det_boxes = _apply_detector_mask(
                    img_color,
                    conf=DEFAULT_MASK_DETECTOR_CONF,
                    imgsz=DEFAULT_MASK_DETECTOR_IMGSZ,
                    allowed_classes=allowed,
                    sat_min=DEFAULT_MASK_COLOR_SAT_MIN,
                    val_min=DEFAULT_MASK_COLOR_VAL_MIN,
                    text_thr=DEFAULT_MASK_TEXT_THR,
                    text_dilate=DEFAULT_MASK_TEXT_DILATE,
                    morph=DEFAULT_MASK_MORPH,
                    mask_dilate=mask_dilate,
                    strategy=DEFAULT_MASK_STRATEGY,
                    text_density_thr=DEFAULT_MASK_TEXT_DENSITY,
                    blur_k=DEFAULT_MASK_BLUR_K,
                    inpaint_radius=DEFAULT_MASK_INPAINT_RADIUS,
                    inpaint_method=DEFAULT_MASK_INPAINT_METHOD,
                )
                if det_img is not None:
                    img = det_img
                    used_detector = True
            except Exception:
                used_detector = False
        if not used_detector:
            img = _mask_stamps_and_signatures(
                img_color,
                mask_stamps=mask_stamps,
                mask_signatures=mask_signatures,
                stamp_min_area=stamp_min_area,
                stamp_max_area=stamp_max_area,
                stamp_circularity=stamp_circularity,
                stamp_rect_aspect_min=stamp_rect_aspect_min,
                stamp_rect_aspect_max=stamp_rect_aspect_max,
                signature_region=signature_region,
                mask_dilate=mask_dilate,
            )
        if detected_doc is not None:
            vis = img_color.copy()
            if det_boxes:
                for box in det_boxes:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    label = box.get("label") or ""
                    score = box.get("conf")
                    color = (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {score:.2f}" if score is not None else label
                    if text:
                        cv2.putText(
                            vis,
                            text,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vis_bytes = cv2.imencode(".png", vis_rgb)[1].tobytes()
            rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
            det_page = detected_doc.new_page(width=page.rect.width, height=page.rect.height)
            det_page.insert_image(rect, stream=vis_bytes)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(rect, stream=img_bytes)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(output_pdf)
    out_doc.close()
    if detected_doc is not None and detected_pdf is not None:
        detected_pdf.parent.mkdir(parents=True, exist_ok=True)
        detected_doc.save(detected_pdf)
        detected_doc.close()
    doc.close()
    return output_pdf


def _ocr_searchable_cpu(
    input_pdf: Path,
    output_pdf: Path,
    lang: str,
    *,
    deskew: bool,
    clean: bool,
    remove_vectors: bool,
    psm: str | None,
    jobs: int | None,
) -> None:
    # OCRmyPDF uses Tesseract under the hood (CPU). We keep optimize=0 to avoid recompression.
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize", "0",
        "-l", lang,
    ]
    if jobs:
        cmd.extend(["--jobs", str(jobs)])
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


def _ocr_paddle_text(
    input_pdf: Path,
    *,
    dpi: int,
    lang: str,
    out_dir: Path,
    token: str,
) -> tuple[str, Path, Path]:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PaddleOCR is not installed in the service") from exc

    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    doc = fitz.open(input_pdf)
    all_text = []
    pages = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        result = ocr.ocr(img, cls=True)
        page_items = []
        if result:
            for line in result[0]:
                box, (text, conf) = line
                page_items.append(
                    {
                        "box": box,
                        "text": text,
                        "conf": float(conf),
                    }
                )
                all_text.append(text)
        pages.append({"page": i, "items": page_items})
    doc.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{token}_paddle.json"
    text_path = out_dir / f"{token}_paddle.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"pages": pages}, f, ensure_ascii=False, indent=2)
    with text_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(all_text).strip())
    return "\n".join(all_text).strip(), json_path, text_path


def _paddle_masked_searchable_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    dpi: int,
    lang: str,
) -> Path:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PaddleOCR is not installed in the service") from exc

    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    doc = fitz.open(input_pdf)
    out = fitz.open()
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        result = ocr.ocr(img, cls=True)

        rect = page.rect
        new_page = out.new_page(width=rect.width, height=rect.height)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()
        new_page.insert_image(rect, stream=img_bytes)

        if result and result[0]:
            scale_x = rect.width / float(img.shape[1])
            scale_y = rect.height / float(img.shape[0])
            for line in result[0]:
                box, (text, conf) = line
                if not text:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x0, x1 = min(xs) * scale_x, max(xs) * scale_x
                y0, y1 = min(ys) * scale_y, max(ys) * scale_y
                bbox = fitz.Rect(x0, y0, x1, y1)
                font_size = max(6.0, (y1 - y0) * 0.8)
                inserted = 0
                for _ in range(5):
                    inserted = new_page.insert_textbox(
                        bbox,
                        text,
                        fontsize=font_size,
                        fontname="helv",
                        render_mode=3,  # invisible text
                        align=0,
                    )
                    if inserted > 0:
                        break
                    font_size = max(4.0, font_size * 0.8)
                if inserted == 0:
                    new_page.insert_text(
                        fitz.Point(x0, y1),
                        text,
                        fontsize=max(4.0, font_size),
                        fontname="helv",
                        render_mode=3,
                    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_pdf)
    out.close()
    doc.close()
    return output_pdf


def _paddle_searchable_from_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    dpi: int,
    lang: str,
) -> Path:
    return _paddle_masked_searchable_pdf(input_pdf, output_pdf, dpi=dpi, lang=lang)


def _strip_images_inplace(doc: fitz.Document) -> None:
    for page in doc:
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            try:
                page.delete_image(xref)
            except Exception:
                continue


def _merge_ocr_layer(
    original_pdf: Path,
    ocr_pdf: Path,
    output_pdf: Path,
) -> Path:
    orig = fitz.open(original_pdf)
    ocr = fitz.open(ocr_pdf)
    _strip_images_inplace(ocr)
    out = fitz.open()
    try:
        page_count = min(len(orig), len(ocr))
        for i in range(page_count):
            orig_page = orig.load_page(i)
            rect = orig_page.rect
            new_page = out.new_page(width=rect.width, height=rect.height)
            new_page.show_pdf_page(rect, orig, i)
            new_page.show_pdf_page(rect, ocr, i, overlay=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_pdf)
    finally:
        out.close()
        ocr.close()
        orig.close()
    return output_pdf


def _extract_ocr_layer_pdf(
    input_pdf: Path,
    output_pdf: Path,
) -> Path:
    doc = fitz.open(input_pdf)
    try:
        _strip_images_inplace(doc)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_pdf)
    finally:
        doc.close()
    return output_pdf


def _insert_invisible_words(page: fitz.Page, words: list[tuple]) -> None:
    for word in words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, text = word[:5]
        if not text:
            continue
        bbox = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        font_size = max(4.0, min(18.0, bbox.height * 0.8))
        inserted = 0
        for _ in range(5):
            inserted = page.insert_textbox(
                bbox,
                str(text),
                fontsize=font_size,
                fontname="helv",
                render_mode=3,
                align=0,
            )
            if inserted > 0:
                break
            font_size = max(4.0, font_size * 0.8)
        if inserted == 0:
            page.insert_text(
                fitz.Point(bbox.x0, bbox.y1),
                str(text),
                fontsize=font_size,
                fontname="helv",
                render_mode=3,
            )


def _line_group_key(word: tuple) -> tuple[int, int, int] | None:
    if len(word) >= 8:
        return (int(word[5]), int(word[6]), 0)
    return None


def _insert_invisible_text_lines(page: fitz.Page, words: list[tuple]) -> None:
    if not words:
        return

    keyed_groups: dict[tuple[int, int, int], list[tuple]] = {}
    unkeyed: list[tuple] = []
    for word in words:
        key = _line_group_key(word)
        if key is None:
            unkeyed.append(word)
        else:
            keyed_groups.setdefault(key, []).append(word)

    line_groups = list(keyed_groups.values())
    if unkeyed:
        unkeyed_sorted = sorted(unkeyed, key=lambda w: (float(w[1]), float(w[0])))
        current: list[tuple] = []
        current_center = None
        current_height = None
        for word in unkeyed_sorted:
            rect = _word_rect(word)
            if rect is None:
                continue
            center = (rect.y0 + rect.y1) / 2.0
            height = rect.height
            if not current:
                current = [word]
                current_center = center
                current_height = height
                continue
            tol = max(3.0, (current_height or height) * 0.6)
            if abs(center - (current_center or center)) <= tol:
                current.append(word)
                current_center = ((current_center or center) * (len(current) - 1) + center) / len(current)
                current_height = max(current_height or 0.0, height)
            else:
                line_groups.append(current)
                current = [word]
                current_center = center
                current_height = height
        if current:
            line_groups.append(current)

    ordered_groups: list[list[tuple]] = []
    for group in line_groups:
        clean_group = [w for w in group if _word_rect(w) is not None and len(w) >= 5 and str(w[4]).strip()]
        if clean_group:
            clean_group.sort(key=lambda w: float(w[0]))
            ordered_groups.append(clean_group)
    ordered_groups.sort(key=lambda g: min(float(w[1]) for w in g))

    for group in ordered_groups:
        rects = [_word_rect(w) for w in group]
        rects = [r for r in rects if r is not None]
        if not rects:
            continue
        line_bbox = fitz.Rect(
            min(r.x0 for r in rects),
            min(r.y0 for r in rects),
            max(r.x1 for r in rects),
            max(r.y1 for r in rects),
        )
        text = " ".join(str(w[4]).strip() for w in group if len(w) >= 5 and str(w[4]).strip())
        if not text:
            continue
        font_size = max(4.0, min(18.0, line_bbox.height * 0.8))
        inserted = 0
        for _ in range(5):
            inserted = page.insert_textbox(
                line_bbox,
                text,
                fontsize=font_size,
                fontname="helv",
                render_mode=3,
                align=0,
            )
            if inserted > 0:
                break
            font_size = max(4.0, font_size * 0.8)
        if inserted == 0:
            page.insert_text(
                fitz.Point(line_bbox.x0, line_bbox.y1),
                text,
                fontsize=font_size,
                fontname="helv",
                render_mode=3,
            )


def _word_rect(word: tuple) -> fitz.Rect | None:
    if len(word) < 4:
        return None
    x0, y0, x1, y1 = word[:4]
    rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
    if rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _rect_intersection_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = a.width * a.height
    if area <= 0:
        return 0.0
    return inter / area


def _word_intersects_regions(
    word: tuple,
    regions: list[fitz.Rect],
    *,
    min_ratio: float = 0.2,
) -> bool:
    rect = _word_rect(word)
    if rect is None or not regions:
        return False
    center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
    for region in regions:
        if region.contains(center):
            return True
        if _rect_intersection_ratio(rect, region) >= min_ratio:
            return True
    return False


def _redact_ocr_layer_by_detector(
    source_pdf: Path,
    ocr_pdf: Path,
    output_pdf: Path,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
    shrink_factor: float = 0.78,
) -> Path:
    source = fitz.open(source_pdf)
    doc = fitz.open(ocr_pdf)
    out = fitz.open()
    try:
        _strip_images_inplace(doc)
        page_count = min(len(source), len(doc))
        for i in range(page_count):
            source_page = source.load_page(i)
            page = doc.load_page(i)
            words = page.get_text("words")
            word_rects = _word_boxes_as_rects(words)
            blocked = _page_detector_boxes_in_pdf_coords(
                source_page,
                conf=conf,
                imgsz=imgsz,
                allowed_classes=allowed_classes,
            )
            text_rects, _text_model_name = _page_text_boxes_in_pdf_coords(
                source_page,
                conf=DEFAULT_TEXT_BLOCK_CONF,
                imgsz=DEFAULT_TEXT_BLOCK_IMGSZ,
            )
            overlap_rects = text_rects or word_rects
            for item in blocked:
                rect = item["rect"]
                redactions, _segments = _build_ocr_redaction_rects(
                    source_page.rect,
                    rect,
                    text_rects=overlap_rects,
                    shrink_factor=shrink_factor,
                    detector_label=item.get("label") or "",
                )
                for redaction in redactions:
                    page.add_redact_annot(redaction, fill=(1, 1, 1))
            if blocked:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            kept_words = page.get_text("words")
            new_page = out.new_page(width=source_page.rect.width, height=source_page.rect.height)
            _insert_invisible_words(new_page, kept_words)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_pdf)
    finally:
        out.close()
        doc.close()
        source.close()
    return output_pdf


def _page_detector_boxes_in_pdf_coords(
    page: fitz.Page,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
) -> list[dict]:
    mat = fitz.Matrix(DEFAULT_MASK_DPI / 72, DEFAULT_MASK_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    boxes = _detect_stamp_boxes(
        img,
        conf=conf,
        imgsz=imgsz,
        allowed_classes=allowed_classes,
    )
    if not boxes:
        return []

    sx = page.rect.width / float(img.shape[1])
    sy = page.rect.height / float(img.shape[0])
    rects: list[dict] = []
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        rects.append(
            {
                "rect": fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                "label": box.get("label") or "",
                "conf": box.get("conf"),
            }
        )
    return rects


def _page_text_boxes_in_pdf_coords(
    page: fitz.Page,
    *,
    conf: float,
    imgsz: int,
) -> tuple[list[fitz.Rect], str | None]:
    mat = fitz.Matrix(DEFAULT_MASK_DPI / 72, DEFAULT_MASK_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    boxes, model_name = _detect_text_block_boxes(
        img,
        conf=conf,
        imgsz=imgsz,
    )
    if not boxes:
        return [], model_name

    sx = page.rect.width / float(img.shape[1])
    sy = page.rect.height / float(img.shape[0])
    rects: list[fitz.Rect] = []
    for box in boxes:
        rects.append(
            fitz.Rect(
                box["x1"] * sx,
                box["y1"] * sy,
                box["x2"] * sx,
                box["y2"] * sy,
            )
        )
    return rects, model_name


def _word_boxes_as_rects(words: list[tuple]) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for word in words:
        x0, y0, x1, y1 = word[:4]
        rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects


def _rects_overlap_side(det_rect: fitz.Rect, rects: list[fitz.Rect], *, side: str, margin_x: float, margin_y: float) -> bool:
    for bbox in rects:
        if side == "left":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x0 < det_rect.x0 + margin_x and bbox.x1 > det_rect.x0:
                return True
        elif side == "right":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x1 > det_rect.x1 - margin_x and bbox.x0 < det_rect.x1:
                return True
        elif side == "top":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y0 < det_rect.y0 + margin_y and bbox.y1 > det_rect.y0:
                return True
        elif side == "bottom":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y1 > det_rect.y1 - margin_y and bbox.y0 < det_rect.y1:
                return True
    return False


def _rect_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)


def _merge_segments(segments: list[tuple[float, float]], *, gap: float = 0.0) -> list[tuple[float, float]]:
    if not segments:
        return []
    ordered = sorted((min(a, b), max(a, b)) for a, b in segments if b > a)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        cur = merged[-1]
        if start <= cur[1] + gap:
            cur[1] = max(cur[1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _side_contact_segments(
    det_rect: fitz.Rect,
    text_rects: list[fitz.Rect],
    *,
    side: str,
    margin_x: float,
    margin_y: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for bbox in text_rects:
        if side == "left":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x0 < det_rect.x0 + margin_x and bbox.x1 > det_rect.x0:
                segments.append((max(det_rect.y0, bbox.y0), min(det_rect.y1, bbox.y1)))
        elif side == "right":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x1 > det_rect.x1 - margin_x and bbox.x0 < det_rect.x1:
                segments.append((max(det_rect.y0, bbox.y0), min(det_rect.y1, bbox.y1)))
        elif side == "top":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y0 < det_rect.y0 + margin_y and bbox.y1 > det_rect.y0:
                segments.append((max(det_rect.x0, bbox.x0), min(det_rect.x1, bbox.x1)))
        elif side == "bottom":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y1 > det_rect.y1 - margin_y and bbox.y0 < det_rect.y1:
                segments.append((max(det_rect.x0, bbox.x0), min(det_rect.x1, bbox.x1)))
    return _merge_segments(segments, gap=2.0)


def _merge_adjacent_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    merged = [fitz.Rect(r) for r in rects if r.width > 0 and r.height > 0]
    changed = True
    eps = 0.01
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        while merged:
            cur = merged.pop(0)
            merged_with_cur = False
            for i, other in enumerate(merged):
                same_x = abs(cur.x0 - other.x0) <= eps and abs(cur.x1 - other.x1) <= eps
                touch_y = abs(cur.y1 - other.y0) <= eps or abs(other.y1 - cur.y0) <= eps
                overlap_y = min(cur.y1, other.y1) - max(cur.y0, other.y0)
                same_y = abs(cur.y0 - other.y0) <= eps and abs(cur.y1 - other.y1) <= eps
                touch_x = abs(cur.x1 - other.x0) <= eps or abs(other.x1 - cur.x0) <= eps
                overlap_x = min(cur.x1, other.x1) - max(cur.x0, other.x0)
                if same_x and (touch_y or overlap_y >= -eps):
                    cur = fitz.Rect(cur.x0, min(cur.y0, other.y0), cur.x1, max(cur.y1, other.y1))
                    merged.pop(i)
                    merged_with_cur = True
                    changed = True
                    break
                if same_y and (touch_x or overlap_x >= -eps):
                    cur = fitz.Rect(min(cur.x0, other.x0), cur.y0, max(cur.x1, other.x1), cur.y1)
                    merged.pop(i)
                    merged_with_cur = True
                    changed = True
                    break
            if merged_with_cur:
                merged.insert(0, cur)
            else:
                out.append(cur)
        merged = out
    return merged


def _shrink_rect(rect: fitz.Rect, factor: float) -> fitz.Rect:
    factor = max(0.1, min(1.0, float(factor)))
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    half_w = rect.width * factor / 2.0
    half_h = rect.height * factor / 2.0
    return fitz.Rect(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _side_max_invasion(
    det_rect: fitz.Rect,
    text_rects: list[fitz.Rect],
    *,
    side: str,
    margin_x: float,
    margin_y: float,
) -> float:
    max_depth = 0.0
    for bbox in text_rects:
        if side == "left":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x0 < det_rect.x0 + margin_x and bbox.x1 > det_rect.x0:
                depth = max(0.0, min(det_rect.x1, bbox.x1) - det_rect.x0)
                max_depth = max(max_depth, depth)
        elif side == "right":
            if bbox.y1 <= det_rect.y0 or bbox.y0 >= det_rect.y1:
                continue
            if bbox.x1 > det_rect.x1 - margin_x and bbox.x0 < det_rect.x1:
                depth = max(0.0, det_rect.x1 - max(det_rect.x0, bbox.x0))
                max_depth = max(max_depth, depth)
        elif side == "top":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y0 < det_rect.y0 + margin_y and bbox.y1 > det_rect.y0:
                depth = max(0.0, min(det_rect.y1, bbox.y1) - det_rect.y0)
                max_depth = max(max_depth, depth)
        elif side == "bottom":
            if bbox.x1 <= det_rect.x0 or bbox.x0 >= det_rect.x1:
                continue
            if bbox.y1 > det_rect.y1 - margin_y and bbox.y0 < det_rect.y1:
                depth = max(0.0, det_rect.y1 - max(det_rect.y0, bbox.y0))
                max_depth = max(max_depth, depth)
    return max_depth


def _build_ocr_redaction_rects_fixed(
    page_rect: fitz.Rect,
    det_rect: fitz.Rect,
    *,
    text_rects: list[fitz.Rect],
    shrink_factor: float,
) -> tuple[list[fitz.Rect], dict[str, list[tuple[float, float]]]]:
    margin_x = max(5.0, det_rect.width * (1.0 - shrink_factor))
    margin_y = max(5.0, det_rect.height * (1.0 - shrink_factor))
    segments = {
        "left": _side_contact_segments(det_rect, text_rects, side="left", margin_x=margin_x, margin_y=margin_y),
        "right": _side_contact_segments(det_rect, text_rects, side="right", margin_x=margin_x, margin_y=margin_y),
        "top": _side_contact_segments(det_rect, text_rects, side="top", margin_x=margin_x, margin_y=margin_y),
        "bottom": _side_contact_segments(det_rect, text_rects, side="bottom", margin_x=margin_x, margin_y=margin_y),
    }

    x0 = det_rect.x0 + (margin_x if segments["left"] else 0.0)
    x1 = det_rect.x1 - (margin_x if segments["right"] else 0.0)
    y0 = det_rect.y0 + (margin_y if segments["top"] else 0.0)
    y1 = det_rect.y1 - (margin_y if segments["bottom"] else 0.0)
    if x1 <= x0:
        cx = (det_rect.x0 + det_rect.x1) / 2.0
        half = max(2.0, margin_x / 2.0)
        x0 = max(det_rect.x0, cx - half)
        x1 = min(det_rect.x1, cx + half)
    if y1 <= y0:
        cy = (det_rect.y0 + det_rect.y1) / 2.0
        half = max(2.0, margin_y / 2.0)
        y0 = max(det_rect.y0, cy - half)
        y1 = min(det_rect.y1, cy + half)
    return [fitz.Rect(
        max(page_rect.x0, x0),
        max(page_rect.y0, y0),
        min(page_rect.x1, x1),
        min(page_rect.y1, y1),
    )], segments


def _build_ocr_redaction_rects(
    page_rect: fitz.Rect,
    det_rect: fitz.Rect,
    *,
    text_rects: list[fitz.Rect],
    shrink_factor: float,
    detector_label: str = "",
) -> tuple[list[fitz.Rect], dict[str, list[tuple[float, float]]]]:
    if detector_label != "sello_redondo":
        return _build_ocr_redaction_rects_fixed(
            page_rect,
            det_rect,
            text_rects=text_rects,
            shrink_factor=shrink_factor,
        )

    margin_x = max(5.0, det_rect.width * (1.0 - shrink_factor))
    margin_y = max(5.0, det_rect.height * (1.0 - shrink_factor))
    keep_left = min(margin_x, max(1.0, _side_max_invasion(det_rect, text_rects, side="left", margin_x=margin_x, margin_y=margin_y) + 1.0))
    keep_right = min(margin_x, max(1.0, _side_max_invasion(det_rect, text_rects, side="right", margin_x=margin_x, margin_y=margin_y) + 1.0))
    keep_top = min(margin_y, max(1.0, _side_max_invasion(det_rect, text_rects, side="top", margin_x=margin_x, margin_y=margin_y) + 1.0))
    keep_bottom = min(margin_y, max(1.0, _side_max_invasion(det_rect, text_rects, side="bottom", margin_x=margin_x, margin_y=margin_y) + 1.0))
    segments = {
        "left": _side_contact_segments(det_rect, text_rects, side="left", margin_x=margin_x, margin_y=margin_y),
        "right": _side_contact_segments(det_rect, text_rects, side="right", margin_x=margin_x, margin_y=margin_y),
        "top": _side_contact_segments(det_rect, text_rects, side="top", margin_x=margin_x, margin_y=margin_y),
        "bottom": _side_contact_segments(det_rect, text_rects, side="bottom", margin_x=margin_x, margin_y=margin_y),
    }

    keep_rects: list[fitz.Rect] = []
    for y0, y1 in segments["left"]:
        keep_rects.append(fitz.Rect(det_rect.x0, y0, min(det_rect.x1, det_rect.x0 + keep_left), y1))
    for y0, y1 in segments["right"]:
        keep_rects.append(fitz.Rect(max(det_rect.x0, det_rect.x1 - keep_right), y0, det_rect.x1, y1))
    for x0, x1 in segments["top"]:
        keep_rects.append(fitz.Rect(x0, det_rect.y0, x1, min(det_rect.y1, det_rect.y0 + keep_top)))
    for x0, x1 in segments["bottom"]:
        keep_rects.append(fitz.Rect(x0, max(det_rect.y0, det_rect.y1 - keep_bottom), x1, det_rect.y1))

    keep_rects = [r for r in keep_rects if r.width > 0 and r.height > 0]
    if not keep_rects:
        return [fitz.Rect(
            max(page_rect.x0, det_rect.x0),
            max(page_rect.y0, det_rect.y0),
            min(page_rect.x1, det_rect.x1),
            min(page_rect.y1, det_rect.y1),
        )], segments

    xs = sorted({det_rect.x0, det_rect.x1, *[r.x0 for r in keep_rects], *[r.x1 for r in keep_rects]})
    ys = sorted({det_rect.y0, det_rect.y1, *[r.y0 for r in keep_rects], *[r.y1 for r in keep_rects]})
    redactions: list[fitz.Rect] = []
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            x0, x1 = xs[xi], xs[xi + 1]
            y0, y1 = ys[yi], ys[yi + 1]
            if x1 <= x0 or y1 <= y0:
                continue
            cell = fitz.Rect(x0, y0, x1, y1)
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            if not det_rect.contains(fitz.Point(mx, my)):
                continue
            if any(r.contains(fitz.Point(mx, my)) for r in keep_rects):
                continue
            clipped = fitz.Rect(
                max(page_rect.x0, cell.x0),
                max(page_rect.y0, cell.y0),
                min(page_rect.x1, cell.x1),
                min(page_rect.y1, cell.y1),
            )
            if clipped.width > 0 and clipped.height > 0:
                redactions.append(clipped)
    return _merge_adjacent_rects(redactions), segments


def _detect_box_text_overlap_sides(
    det_rect: fitz.Rect,
    text_rects: list[fitz.Rect],
    *,
    shrink_factor: float,
) -> dict[str, bool]:
    _rects, segments = _build_ocr_redaction_rects(
        fitz.Rect(det_rect),
        det_rect,
        text_rects=text_rects,
        shrink_factor=shrink_factor,
    )
    return {k: bool(v) for k, v in segments.items()}


def _render_overlap_debug_pdf(
    source_pdf: Path,
    output_pdf: Path,
    *,
    conf: float,
    imgsz: int,
    allowed_classes: set[str] | None,
    shrink_factor: float,
) -> Path:
    source = fitz.open(source_pdf)
    out = fitz.open()
    try:
        for i in range(len(source)):
            page = source.load_page(i)
            word_rects = _word_boxes_as_rects(page.get_text("words"))
            blocked = _page_detector_boxes_in_pdf_coords(
                page,
                conf=conf,
                imgsz=imgsz,
                allowed_classes=allowed_classes,
            )
            text_rects, text_model_name = _page_text_boxes_in_pdf_coords(
                page,
                conf=DEFAULT_TEXT_BLOCK_CONF,
                imgsz=DEFAULT_TEXT_BLOCK_IMGSZ,
            )
            overlap_rects = text_rects or word_rects
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.show_pdf_page(page.rect, source, i)

            for text_rect in text_rects:
                new_page.draw_rect(text_rect, color=(0.0, 0.65, 0.0), width=0.8)

            if text_rects:
                new_page.insert_text(
                    fitz.Point(12, 14),
                    f"text_model={text_model_name or 'unknown'}",
                    fontsize=8,
                    color=(0.0, 0.5, 0.0),
                )
            else:
                new_page.insert_text(
                    fitz.Point(12, 14),
                    "text_source=ocr_words_fallback",
                    fontsize=8,
                    color=(0.5, 0.2, 0.0),
                )

            for item in blocked:
                rect = item["rect"]
                redactions, segments = _build_ocr_redaction_rects(
                    page.rect,
                    rect,
                    text_rects=overlap_rects,
                    shrink_factor=shrink_factor,
                    detector_label=item.get("label") or "",
                )
                sides = {k: bool(v) for k, v in segments.items()}

                new_page.draw_rect(rect, color=(1, 0, 0), width=1.5)
                for redaction in redactions:
                    new_page.draw_rect(redaction, color=(0, 0.5, 1), width=1.0)
                label = (
                    f"L={int(sides['left'])} "
                    f"R={int(sides['right'])} "
                    f"T={int(sides['top'])} "
                    f"B={int(sides['bottom'])}"
                )
                new_page.insert_text(
                    fitz.Point(rect.x0, max(10, rect.y0 - 4)),
                    label,
                    fontsize=8,
                    color=(1, 0, 0),
                )

                side_margin_x = max(5.0, rect.width * (1.0 - shrink_factor))
                side_margin_y = max(5.0, rect.height * (1.0 - shrink_factor))
                x_left = min(rect.x1, rect.x0 + side_margin_x)
                x_right = max(rect.x0, rect.x1 - side_margin_x)
                y_top = min(rect.y1, rect.y0 + side_margin_y)
                y_bottom = max(rect.y0, rect.y1 - side_margin_y)
                for y0, y1 in segments["left"]:
                    new_page.draw_line(fitz.Point(x_left, y0), fitz.Point(x_left, y1), color=(1, 0.5, 0), width=1)
                for y0, y1 in segments["right"]:
                    new_page.draw_line(fitz.Point(x_right, y0), fitz.Point(x_right, y1), color=(1, 0.5, 0), width=1)
                for x0, x1 in segments["top"]:
                    new_page.draw_line(fitz.Point(x0, y_top), fitz.Point(x1, y_top), color=(1, 0.5, 0), width=1)
                for x0, x1 in segments["bottom"]:
                    new_page.draw_line(fitz.Point(x0, y_bottom), fitz.Point(x1, y_bottom), color=(1, 0.5, 0), width=1)
                if item.get("label"):
                    new_page.insert_text(
                        fitz.Point(rect.x0, min(page.rect.y1 - 2, rect.y1 + 10)),
                        item["label"],
                        fontsize=7,
                        color=(0.6, 0.0, 0.0),
                    )

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_pdf)
    finally:
        out.close()
        source.close()
    return output_pdf


def run_ocr(
    url: str,
    mode: str | None = None,
    lang: str | None = None,
    *,
    deskew: bool | None = None,
    clean: bool | None = None,
    remove_vectors: bool | None = None,
    psm: str | None = None,
    jobs: int | None = None,
    mask_stamps: bool | None = None,
    mask_signatures: bool | None = None,
    mask_grayscale: bool | None = None,
    mask_dilate: int | None = None,
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
    jobs = DEFAULT_OCR_JOBS if jobs is None else jobs
    mask_stamps = DEFAULT_MASK_STAMPS if mask_stamps is None else mask_stamps
    mask_signatures = DEFAULT_MASK_SIGNATURES if mask_signatures is None else mask_signatures
    mask_grayscale = DEFAULT_MASK_GRAYSCALE if mask_grayscale is None else mask_grayscale
    mask_dilate = DEFAULT_MASK_DILATE if mask_dilate is None else mask_dilate
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
        masked_pdf = None
        masked_searchable_pdf = None
        detected_pdf = None
        if mask_stamps or mask_signatures:
            masked_pdf = out_dir / f"{token}_masked.pdf"
            if DEFAULT_MASK_SAVE_DETECTIONS:
                detected_pdf = out_dir / f"{token}_detected.pdf"
            _prepare_masked_pdf(
                src_pdf,
                masked_pdf,
                mask_stamps=mask_stamps,
                mask_signatures=mask_signatures,
                stamp_min_area=stamp_min_area,
                stamp_max_area=stamp_max_area,
                stamp_circularity=stamp_circularity,
                stamp_rect_aspect_min=stamp_rect_aspect_min,
                stamp_rect_aspect_max=stamp_rect_aspect_max,
                signature_region=signature_region,
                grayscale=mask_grayscale,
                mask_dilate=mask_dilate,
                detected_pdf=detected_pdf,
            )
            masked_searchable_pdf = out_dir / f"{token}_masked_searchable.pdf"
            _ocr_searchable_cpu(
                masked_pdf,
                masked_searchable_pdf,
                lang,
                deskew=deskew,
                clean=clean,
                remove_vectors=remove_vectors,
                psm=psm,
                jobs=jobs,
            )
            paddle_masked_searchable_pdf = None
            try:
                paddle_masked_searchable_pdf = out_dir / f"{token}_paddle_masked_searchable.pdf"
                _paddle_masked_searchable_pdf(
                    masked_pdf,
                    paddle_masked_searchable_pdf,
                    dpi=DEFAULT_PADDLE_DPI,
                    lang=DEFAULT_PADDLE_LANG,
                )
            except Exception:
                paddle_masked_searchable_pdf = None
            _merge_ocr_layer(src_pdf, masked_searchable_pdf, out_pdf)
        else:
            _ocr_searchable_cpu(
                src_pdf,
                out_pdf,
                lang,
                deskew=deskew,
                clean=clean,
                remove_vectors=remove_vectors,
                psm=psm,
                jobs=jobs,
            )
        paddle_searchable_pdf = None
        try:
            paddle_searchable_pdf = out_dir / f"{token}_paddle_searchable.pdf"
            _paddle_searchable_from_pdf(
                src_pdf,
                paddle_searchable_pdf,
                dpi=DEFAULT_PADDLE_DPI,
                lang=DEFAULT_PADDLE_LANG,
            )
        except Exception:
            paddle_searchable_pdf = None
        text = _extract_text(out_pdf)
    elif mode == "searchable_conservative":
        out_pdf = out_dir / f"{token}_searchable.pdf"
        working_pdf = out_dir / f"{token}_working.pdf"
        original_searchable_pdf = out_dir / f"{token}_original_searchable.pdf"
        working_searchable_pdf = out_dir / f"{token}_working_searchable.pdf"
        detected_pdf = out_dir / f"{token}_detected.pdf" if DEFAULT_MASK_SAVE_DETECTIONS else None
        allowed_all = set(DEFAULT_MASK_DETECTOR_CLASSES) if DEFAULT_MASK_DETECTOR_CLASSES else None
        allowed_round = {"sello_redondo"}
        if allowed_all is not None:
            allowed_round = allowed_round & allowed_all
        _prepare_conservative_working_pdf(
            src_pdf,
            working_pdf,
            conf=DEFAULT_MASK_DETECTOR_CONF,
            imgsz=DEFAULT_MASK_DETECTOR_IMGSZ,
            allowed_classes=allowed_round or None,
            grayscale=mask_grayscale,
            detected_pdf=detected_pdf,
        )
        _ocr_searchable_cpu(
            src_pdf,
            original_searchable_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
        )
        _ocr_searchable_cpu(
            working_pdf,
            working_searchable_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
        )
        filtered_layer_pdf = out_dir / f"{token}_filtered_layer.pdf"
        overlap_debug_pdf = out_dir / f"{token}_overlap_debug.pdf"
        overlap_debug_searchable_pdf = out_dir / f"{token}_overlap_debug_searchable.pdf"
        overlap_debug_working_searchable_pdf = out_dir / f"{token}_overlap_debug_working_searchable.pdf"
        _render_overlap_debug_pdf(
            src_pdf,
            overlap_debug_pdf,
            conf=DEFAULT_MASK_DETECTOR_CONF,
            imgsz=DEFAULT_MASK_DETECTOR_IMGSZ,
            allowed_classes=allowed_all,
            shrink_factor=0.78,
        )
        _extract_ocr_layer_pdf(working_searchable_pdf, filtered_layer_pdf)
        _merge_ocr_layer(src_pdf, filtered_layer_pdf, out_pdf)
        _merge_ocr_layer(overlap_debug_pdf, filtered_layer_pdf, overlap_debug_searchable_pdf)
        _merge_ocr_layer(overlap_debug_pdf, working_searchable_pdf, overlap_debug_working_searchable_pdf)
        masked_pdf = working_pdf
        masked_searchable_pdf = None
        paddle_masked_searchable_pdf = None
        paddle_searchable_pdf = None
        text = _extract_text(out_pdf)
    elif mode == "paddle_text":
        text, paddle_json, paddle_text = _ocr_paddle_text(
            src_pdf,
            dpi=DEFAULT_PADDLE_DPI,
            lang=DEFAULT_PADDLE_LANG,
            out_dir=out_dir,
            token=token,
        )
        out_pdf = None
        masked_pdf = None
        masked_searchable_pdf = None
        detected_pdf = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    elapsed = time.time() - start

    return {
        "mode": mode,
        "lang": lang,
        "source": str(src_pdf),
        "output_pdf": str(out_pdf) if out_pdf else None,
        "masked_pdf": str(masked_pdf) if masked_pdf else None,
        "masked_output_pdf": str(masked_searchable_pdf) if masked_searchable_pdf else None,
        "original_searchable_pdf": str(original_searchable_pdf)
        if "original_searchable_pdf" in locals()
        else None,
        "working_searchable_pdf": str(working_searchable_pdf)
        if "working_searchable_pdf" in locals()
        else None,
        "detected_pdf": str(detected_pdf) if detected_pdf else None,
            "paddle_masked_searchable_pdf": str(paddle_masked_searchable_pdf)
        if "paddle_masked_searchable_pdf" in locals() and paddle_masked_searchable_pdf
        else None,
        "overlap_debug_pdf": str(overlap_debug_pdf)
        if "overlap_debug_pdf" in locals()
        else None,
        "overlap_debug_searchable_pdf": str(overlap_debug_searchable_pdf)
        if "overlap_debug_searchable_pdf" in locals()
        else None,
        "overlap_debug_working_searchable_pdf": str(overlap_debug_working_searchable_pdf)
        if "overlap_debug_working_searchable_pdf" in locals()
        else None,
        "paddle_searchable_pdf": str(paddle_searchable_pdf)
        if "paddle_searchable_pdf" in locals() and paddle_searchable_pdf
        else None,
        "paddle_json": str(paddle_json) if mode == "paddle_text" else None,
        "paddle_text": str(paddle_text) if mode == "paddle_text" else None,
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
    jobs: int | None = None,
    mask_stamps: bool | None = None,
    mask_signatures: bool | None = None,
    mask_grayscale: bool | None = None,
    mask_dilate: int | None = None,
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
    jobs = DEFAULT_OCR_JOBS if jobs is None else jobs
    mask_stamps = DEFAULT_MASK_STAMPS if mask_stamps is None else mask_stamps
    mask_signatures = DEFAULT_MASK_SIGNATURES if mask_signatures is None else mask_signatures
    mask_grayscale = DEFAULT_MASK_GRAYSCALE if mask_grayscale is None else mask_grayscale
    mask_dilate = DEFAULT_MASK_DILATE if mask_dilate is None else mask_dilate
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
        masked_pdf = None
        masked_searchable_pdf = None
        detected_pdf = None
        if mask_stamps or mask_signatures:
            masked_pdf = out_dir / f"{token}_masked.pdf"
            if DEFAULT_MASK_SAVE_DETECTIONS:
                detected_pdf = out_dir / f"{token}_detected.pdf"
            _prepare_masked_pdf(
                src_pdf,
                masked_pdf,
                mask_stamps=mask_stamps,
                mask_signatures=mask_signatures,
                stamp_min_area=stamp_min_area,
                stamp_max_area=stamp_max_area,
                stamp_circularity=stamp_circularity,
                stamp_rect_aspect_min=stamp_rect_aspect_min,
                stamp_rect_aspect_max=stamp_rect_aspect_max,
                signature_region=signature_region,
                grayscale=mask_grayscale,
                mask_dilate=mask_dilate,
                detected_pdf=detected_pdf,
            )
            masked_searchable_pdf = out_dir / f"{token}_masked_searchable.pdf"
            _ocr_searchable_cpu(
                masked_pdf,
                masked_searchable_pdf,
                lang,
                deskew=deskew,
                clean=clean,
                remove_vectors=remove_vectors,
                psm=psm,
                jobs=jobs,
            )
            paddle_masked_searchable_pdf = None
            try:
                paddle_masked_searchable_pdf = out_dir / f"{token}_paddle_masked_searchable.pdf"
                _paddle_masked_searchable_pdf(
                    masked_pdf,
                    paddle_masked_searchable_pdf,
                    dpi=DEFAULT_PADDLE_DPI,
                    lang=DEFAULT_PADDLE_LANG,
                )
            except Exception:
                paddle_masked_searchable_pdf = None
            _merge_ocr_layer(src_pdf, masked_searchable_pdf, out_pdf)
        else:
            _ocr_searchable_cpu(
                src_pdf,
                out_pdf,
                lang,
                deskew=deskew,
                clean=clean,
                remove_vectors=remove_vectors,
                psm=psm,
                jobs=jobs,
            )
        paddle_searchable_pdf = None
        try:
            paddle_searchable_pdf = out_dir / f"{token}_paddle_searchable.pdf"
            _paddle_searchable_from_pdf(
                src_pdf,
                paddle_searchable_pdf,
                dpi=DEFAULT_PADDLE_DPI,
                lang=DEFAULT_PADDLE_LANG,
            )
        except Exception:
            paddle_searchable_pdf = None
        text = _extract_text(out_pdf)
    elif mode == "searchable_conservative":
        out_pdf = out_dir / f"{token}_searchable.pdf"
        working_pdf = out_dir / f"{token}_working.pdf"
        original_searchable_pdf = out_dir / f"{token}_original_searchable.pdf"
        working_searchable_pdf = out_dir / f"{token}_working_searchable.pdf"
        detected_pdf = out_dir / f"{token}_detected.pdf" if DEFAULT_MASK_SAVE_DETECTIONS else None
        allowed_all = set(DEFAULT_MASK_DETECTOR_CLASSES) if DEFAULT_MASK_DETECTOR_CLASSES else None
        allowed_round = {"sello_redondo"}
        if allowed_all is not None:
            allowed_round = allowed_round & allowed_all
        _prepare_conservative_working_pdf(
            src_pdf,
            working_pdf,
            conf=DEFAULT_MASK_DETECTOR_CONF,
            imgsz=DEFAULT_MASK_DETECTOR_IMGSZ,
            allowed_classes=allowed_round or None,
            grayscale=mask_grayscale,
            detected_pdf=detected_pdf,
        )
        _ocr_searchable_cpu(
            src_pdf,
            original_searchable_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
        )
        _ocr_searchable_cpu(
            working_pdf,
            working_searchable_pdf,
            lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
        )
        filtered_layer_pdf = out_dir / f"{token}_filtered_layer.pdf"
        overlap_debug_pdf = out_dir / f"{token}_overlap_debug.pdf"
        overlap_debug_searchable_pdf = out_dir / f"{token}_overlap_debug_searchable.pdf"
        overlap_debug_working_searchable_pdf = out_dir / f"{token}_overlap_debug_working_searchable.pdf"
        _render_overlap_debug_pdf(
            src_pdf,
            overlap_debug_pdf,
            conf=DEFAULT_MASK_DETECTOR_CONF,
            imgsz=DEFAULT_MASK_DETECTOR_IMGSZ,
            allowed_classes=allowed_all,
            shrink_factor=0.78,
        )
        _extract_ocr_layer_pdf(working_searchable_pdf, filtered_layer_pdf)
        _merge_ocr_layer(src_pdf, filtered_layer_pdf, out_pdf)
        _merge_ocr_layer(overlap_debug_pdf, filtered_layer_pdf, overlap_debug_searchable_pdf)
        _merge_ocr_layer(overlap_debug_pdf, working_searchable_pdf, overlap_debug_working_searchable_pdf)
        masked_pdf = working_pdf
        masked_searchable_pdf = None
        paddle_masked_searchable_pdf = None
        paddle_searchable_pdf = None
        text = _extract_text(out_pdf)
    elif mode == "paddle_text":
        text, paddle_json, paddle_text = _ocr_paddle_text(
            src_pdf,
            dpi=DEFAULT_PADDLE_DPI,
            lang=DEFAULT_PADDLE_LANG,
            out_dir=out_dir,
            token=token,
        )
        out_pdf = None
        masked_pdf = None
        masked_searchable_pdf = None
        detected_pdf = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    elapsed = time.time() - start

    return {
        "mode": mode,
        "lang": lang,
        "source": str(src_pdf),
        "output_pdf": str(out_pdf) if out_pdf else None,
        "masked_pdf": str(masked_pdf) if masked_pdf else None,
        "masked_output_pdf": str(masked_searchable_pdf) if masked_searchable_pdf else None,
        "original_searchable_pdf": str(original_searchable_pdf)
        if "original_searchable_pdf" in locals()
        else None,
        "working_searchable_pdf": str(working_searchable_pdf)
        if "working_searchable_pdf" in locals()
        else None,
        "detected_pdf": str(detected_pdf) if detected_pdf else None,
            "paddle_masked_searchable_pdf": str(paddle_masked_searchable_pdf)
        if "paddle_masked_searchable_pdf" in locals() and paddle_masked_searchable_pdf
        else None,
        "overlap_debug_pdf": str(overlap_debug_pdf)
        if "overlap_debug_pdf" in locals()
        else None,
        "overlap_debug_searchable_pdf": str(overlap_debug_searchable_pdf)
        if "overlap_debug_searchable_pdf" in locals()
        else None,
        "overlap_debug_working_searchable_pdf": str(overlap_debug_working_searchable_pdf)
        if "overlap_debug_working_searchable_pdf" in locals()
        else None,
        "paddle_searchable_pdf": str(paddle_searchable_pdf)
        if "paddle_searchable_pdf" in locals() and paddle_searchable_pdf
        else None,
        "paddle_json": str(paddle_json) if mode == "paddle_text" else None,
        "paddle_text": str(paddle_text) if mode == "paddle_text" else None,
        "content_type": content_type,
        "source_bytes": size,
        "text_len": len(text),
        "elapsed_sec": round(elapsed, 2),
    }


def run_stamp_test() -> dict:
    pdf_path = Path(DEFAULT_STAMP_TEST_PDF)
    if not pdf_path.exists():
        raise RuntimeError(f"STAMP_TEST_PDF not found: {pdf_path}")

    model_path = Path(DEFAULT_STAMP_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"STAMP_MODEL_PATH not found: {model_path}")

    out_dir = Path(DEFAULT_OUT_DIR) / "stamp_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:12]
    img_path = out_dir / f"{token}_p1.png"

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        mat = fitz.Matrix(DEFAULT_STAMP_TEST_DPI / 72, DEFAULT_STAMP_TEST_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(img_path.as_posix())
    finally:
        doc.close()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ultralytics is not installed in the service") from exc

    model = YOLO(str(model_path))
    device = DEFAULT_STAMP_DEVICE
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    predict_dir = out_dir / "predict"
    model.predict(
        source=str(img_path),
        imgsz=DEFAULT_STAMP_TEST_IMGSZ,
        conf=DEFAULT_STAMP_TEST_CONF,
        device=device,
        save=True,
        verbose=False,
        project=str(predict_dir),
        name=token,
        exist_ok=True,
    )

    pred_dir = predict_dir / token
    output_image = None
    for ext in (".jpg", ".png"):
        matches = list(pred_dir.glob(f"*{ext}"))
        if matches:
            output_image = matches[0]
            break
    if output_image is None:
        raise RuntimeError(f"Prediction image not found in {pred_dir}")

    return {
        "source_pdf": str(pdf_path),
        "source_image": str(img_path),
        "output_image": str(output_image),
    }

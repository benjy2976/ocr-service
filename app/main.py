import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from app.ocr_pipeline import DEFAULT_OUT_DIR, run_ocr, run_ocr_file

app = FastAPI(title="OCR Pilot Service", version="0.1.0")


class OCRRequest(BaseModel):
    url: HttpUrl
    mode: str | None = None
    lang: str | None = None
    deskew: bool | None = None
    clean: bool | None = None
    remove_vectors: bool | None = None
    psm: str | None = None
    jobs: int | None = None
    mask_stamps: bool | None = None
    mask_signatures: bool | None = None
    mask_grayscale: bool | None = None
    mask_dilate: int | None = None
    stamp_min_area: float | None = None
    stamp_max_area: float | None = None
    stamp_circularity: float | None = None
    stamp_rect_aspect_min: float | None = None
    stamp_rect_aspect_max: float | None = None
    signature_region: float | None = None


class OCRLocalRequest(BaseModel):
    path: str
    mode: str | None = None
    lang: str | None = None
    deskew: bool | None = None
    clean: bool | None = None
    remove_vectors: bool | None = None
    psm: str | None = None
    jobs: int | None = None
    mask_stamps: bool | None = None
    mask_signatures: bool | None = None
    mask_grayscale: bool | None = None
    mask_dilate: int | None = None
    stamp_min_area: float | None = None
    stamp_max_area: float | None = None
    stamp_circularity: float | None = None
    stamp_rect_aspect_min: float | None = None
    stamp_rect_aspect_max: float | None = None
    signature_region: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(req: OCRRequest):
    try:
        result = run_ocr(
            url=str(req.url),
            mode=req.mode,
            lang=req.lang,
            deskew=req.deskew,
            clean=req.clean,
            remove_vectors=req.remove_vectors,
            psm=req.psm,
            jobs=req.jobs,
            mask_stamps=req.mask_stamps,
            mask_signatures=req.mask_signatures,
            mask_grayscale=req.mask_grayscale,
            mask_dilate=req.mask_dilate,
            stamp_min_area=req.stamp_min_area,
            stamp_max_area=req.stamp_max_area,
            stamp_circularity=req.stamp_circularity,
            stamp_rect_aspect_min=req.stamp_rect_aspect_min,
            stamp_rect_aspect_max=req.stamp_rect_aspect_max,
            signature_region=req.signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    local_root = os.getenv("OCR_LOCAL_ROOT")
    if local_root:
        root = Path(local_root).expanduser().resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Path outside OCR_LOCAL_ROOT")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="File not found")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return path


@app.post("/ocr/local")
def ocr_local(req: OCRLocalRequest):
    try:
        path = _resolve_local_path(req.path)
        result = run_ocr_file(
            path,
            mode=req.mode,
            lang=req.lang,
            deskew=req.deskew,
            clean=req.clean,
            remove_vectors=req.remove_vectors,
            psm=req.psm,
            jobs=req.jobs,
            mask_stamps=req.mask_stamps,
            mask_signatures=req.mask_signatures,
            mask_grayscale=req.mask_grayscale,
            mask_dilate=req.mask_dilate,
            stamp_min_area=req.stamp_min_area,
            stamp_max_area=req.stamp_max_area,
            stamp_circularity=req.stamp_circularity,
            stamp_rect_aspect_min=req.stamp_rect_aspect_min,
            stamp_rect_aspect_max=req.stamp_rect_aspect_max,
            signature_region=req.signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ocr/file")
def ocr_file(
    file: UploadFile = File(...),
    mode: str | None = Form(None),
    lang: str | None = Form(None),
    deskew: bool | None = Form(None),
    clean: bool | None = Form(None),
    remove_vectors: bool | None = Form(None),
    psm: str | None = Form(None),
    jobs: int | None = Form(None),
    mask_stamps: bool | None = Form(None),
    mask_signatures: bool | None = Form(None),
    mask_grayscale: bool | None = Form(None),
    mask_dilate: int | None = Form(None),
    stamp_min_area: float | None = Form(None),
    stamp_max_area: float | None = Form(None),
    stamp_circularity: float | None = Form(None),
    stamp_rect_aspect_min: float | None = Form(None),
    stamp_rect_aspect_max: float | None = Form(None),
    signature_region: float | None = Form(None),
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        tmp_dir = Path(os.getenv("OCR_TMP_DIR", "/data/tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        safe_name = Path(file.filename).name
        dst = tmp_dir / f"{token}_{safe_name}"
        with dst.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        result = run_ocr_file(
            dst,
            mode=mode,
            lang=lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
            mask_stamps=mask_stamps,
            mask_signatures=mask_signatures,
            mask_grayscale=mask_grayscale,
            mask_dilate=mask_dilate,
            stamp_min_area=stamp_min_area,
            stamp_max_area=stamp_max_area,
            stamp_circularity=stamp_circularity,
            stamp_rect_aspect_min=stamp_rect_aspect_min,
            stamp_rect_aspect_max=stamp_rect_aspect_max,
            signature_region=signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/file/{filename}")
def download_file(filename: str):
    # Prevent path traversal: only allow plain file names
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_dir = Path(DEFAULT_OUT_DIR)
    file_path = out_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), media_type="application/pdf", filename=filename)

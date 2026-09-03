"""API publica de OCR: endpoints genericos, sin logica de revision/clasificacion.

Este modulo es la superficie que un consumidor externo (adapter, otra
plataforma) deberia usar. No debe importar nada de app.worker ni conocer
Normatividad.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, HttpUrl

from app.ocr_pipeline import (
    DEFAULT_OUT_DIR,
    run_ocr,
    run_ocr_file,
    run_stamp_test,
)
from app import jobs as jobs_module

router = APIRouter()


class JobArtifactSpec(BaseModel):
    path: str | None = None
    overwrite: bool = False


class JobRequest(BaseModel):
    source: dict
    options: dict = {}
    artifacts: dict[str, JobArtifactSpec] = {}


@router.post("/jobs", status_code=202)
def create_job(req: JobRequest):
    source_path_str = req.source.get("path") if isinstance(req.source, dict) else None
    if not source_path_str:
        raise HTTPException(status_code=400, detail="source.path is required")
    source_path = _resolve_local_path(source_path_str)

    unknown = set(req.artifacts) - set(jobs_module.ARTIFACT_NAMES)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unsupported artifacts: {sorted(unknown)}")
    if not req.artifacts:
        raise HTTPException(status_code=400, detail="at least one artifact must be requested")

    artifacts_requested = {
        name: {"path": spec.path, "overwrite": spec.overwrite}
        for name, spec in req.artifacts.items()
    }
    job_id = jobs_module.submit_job(source_path, req.options, artifacts_requested)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "artifacts": job["artifacts"],
        "preflight": job["preflight"],
        "error": job["error"],
    }


@router.get("/jobs/{job_id}/artifacts/{name}")
def download_job_artifact(job_id: str, name: str):
    path = jobs_module.get_staged_artifact_path(job_id, name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not available (not staged, not ready, or expired)")
    return FileResponse(path)


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


def _ephemeral_ocr_file_response(output_pdf: Path, temp_root: Path) -> FileResponse:
    return FileResponse(
        path=str(output_pdf),
        media_type="application/pdf",
        background=BackgroundTask(shutil.rmtree, str(temp_root), True),
    )


def _regulations_pdf_url(doc_id: int, tomo: int, num_tipe: int) -> str:
    return f"https://proyectos.regionhuanuco.gob.pe/regulations/file/{doc_id}/{tomo}/{num_tipe}"


def _resolve_download_ocr_mode(mode: str | None) -> str:
    return (mode or "searchable_conservative_service").strip()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ocr")
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


@router.post("/ocr/local")
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


@router.post("/ocr/file")
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


@router.post("/ocr/download")
def ocr_download(req: OCRRequest):
    temp_root = Path(tempfile.mkdtemp(prefix="ocr_ws_", dir=os.getenv("OCR_TMP_DIR", "/data/tmp")))
    req_tmp = temp_root / "tmp"
    req_out = temp_root / "out"
    req_tmp.mkdir(parents=True, exist_ok=True)
    req_out.mkdir(parents=True, exist_ok=True)
    try:
        result = run_ocr(
            url=str(req.url),
            mode=_resolve_download_ocr_mode(req.mode),
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
            tmp_dir=req_tmp,
            out_dir=req_out,
        )
        output_pdf = Path(result["output_pdf"]) if result.get("output_pdf") else None
        if output_pdf is None or not output_pdf.exists():
            raise HTTPException(status_code=500, detail="OCR did not produce a PDF output")
        source_name = Path(str(req.url).split("?")[0]).name or "document.pdf"
        return _ephemeral_ocr_file_response(output_pdf, temp_root)
    except HTTPException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ocr/file/download")
def ocr_file_download(
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
    temp_root = Path(tempfile.mkdtemp(prefix="ocr_ws_", dir=os.getenv("OCR_TMP_DIR", "/data/tmp")))
    req_tmp = temp_root / "tmp"
    req_out = temp_root / "out"
    req_tmp.mkdir(parents=True, exist_ok=True)
    req_out.mkdir(parents=True, exist_ok=True)
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        safe_name = Path(file.filename).name
        src_pdf = req_tmp / safe_name
        with src_pdf.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        result = run_ocr_file(
            src_pdf,
            mode=_resolve_download_ocr_mode(mode),
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
            tmp_dir=req_tmp,
            out_dir=req_out,
        )
        output_pdf = Path(result["output_pdf"]) if result.get("output_pdf") else None
        if output_pdf is None or not output_pdf.exists():
            raise HTTPException(status_code=500, detail="OCR did not produce a PDF output")
        return _ephemeral_ocr_file_response(output_pdf, temp_root)
    except HTTPException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ocr/regulations/{doc_id}/{tomo}/{num_tipe}")
def ocr_regulations_download(
    doc_id: int,
    tomo: int,
    num_tipe: int,
    mode: str | None = None,
    lang: str | None = None,
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
):
    if doc_id <= 0 or tomo < 0 or num_tipe < 0:
        raise HTTPException(status_code=400, detail="Invalid regulations parameters")
    temp_root = Path(tempfile.mkdtemp(prefix="ocr_ws_", dir=os.getenv("OCR_TMP_DIR", "/data/tmp")))
    req_tmp = temp_root / "tmp"
    req_out = temp_root / "out"
    req_tmp.mkdir(parents=True, exist_ok=True)
    req_out.mkdir(parents=True, exist_ok=True)
    try:
        source_url = _regulations_pdf_url(doc_id, tomo, num_tipe)
        result = run_ocr(
            url=source_url,
            mode=_resolve_download_ocr_mode(mode),
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
            tmp_dir=req_tmp,
            out_dir=req_out,
        )
        output_pdf = Path(result["output_pdf"]) if result.get("output_pdf") else None
        if output_pdf is None or not output_pdf.exists():
            raise HTTPException(status_code=500, detail="OCR did not produce a PDF output")
        return _ephemeral_ocr_file_response(output_pdf, temp_root)
    except HTTPException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/file/{filename}")
def download_file(filename: str):
    # Prevent path traversal: only allow plain file names
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_dir = Path(DEFAULT_OUT_DIR)
    file_path = out_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), media_type="application/pdf", filename=filename)


@router.post("/stamps/test")
def stamps_test():
    try:
        result = run_stamp_test()
        output_image = Path(result["output_image"])
        result["output_filename"] = output_image.name
        result["download_path"] = f"/image/{output_image.name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/image/{output_image.name}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/image/{filename}")
def download_image(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    candidates = [
        Path("/data/annotations/images/pages") / filename,
        Path(DEFAULT_OUT_DIR) / filename,
    ]
    file_path = None
    for candidate in candidates:
        if candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        out_dir = Path(DEFAULT_OUT_DIR)
        matches = list(out_dir.glob(f"**/{filename}"))
        if matches:
            file_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="File not found")
    suffix = file_path.suffix.lower()
    if suffix == ".png":
        media_type = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)



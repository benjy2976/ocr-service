"""Motor de jobs asincronos para /jobs.

Contrato aprobado:
  POST /jobs        -> 202, {"job_id": ...}          (no bloquea)
  GET  /jobs/{id}    -> estado + info de artefactos
  GET  /jobs/{id}/artifacts/{name}  -> descarga (solo modo staging, con TTL)

Reglas de diseno:
  - Si la peticion trae `artifacts.<name>.path`, ese es el destino final.
    Debe resolver dentro de una raiz permitida (OCR_ALLOWED_OUTPUT_ROOTS) y
    la escritura es atomica (temporal en el mismo directorio + os.replace).
  - Si no trae destino, el artefacto queda en staging local con TTL y se
    descarga por HTTP una o mas veces dentro de esa ventana.
  - El motor solo escribe a rutas de filesystem montado. Nunca implementa
    clientes de protocolo de almacenamiento (FTP, S3, etc.) - eso es
    responsabilidad de quien monte el volumen en el contenedor.
  - Job store en memoria: valido mientras ocr-api corra con 1 replica
    (hoy: ocr-service/base/05-deployment-api.yaml replicas=1). Si se
    escala a mas replicas, esto debe moverse a un store compartido.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.ocr_pipeline import DEFAULT_OUT_DIR, run_ocr_file
from app.preflight import inspect_pdf
from app.text_artifact import build_text_document

_DEFAULT_STAGING_TTL_MIN = int(os.getenv("OCR_JOB_STAGING_TTL_MIN", "15"))
_STAGING_DIR = Path(os.getenv("OCR_JOB_STAGING_DIR", str(Path(DEFAULT_OUT_DIR) / "job_staging")))
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("OCR_JOB_WORKERS", "2")))

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

ARTIFACT_NAMES = ("pdf", "text")


def _allowed_output_roots() -> list[Path]:
    raw = os.getenv("OCR_ALLOWED_OUTPUT_ROOTS", "").strip()
    roots = [DEFAULT_OUT_DIR]
    shared = os.getenv("OCR_SHARED_CACHE_DIR", "").strip()
    if shared:
        roots.append(shared)
    if raw:
        roots.extend(p.strip() for p in raw.split(",") if p.strip())
    return [Path(r).resolve() for r in roots]


def _resolve_destination(path_str: str) -> Path:
    dest = Path(path_str).resolve()
    for root in _allowed_output_roots():
        try:
            dest.relative_to(root)
            return dest
        except ValueError:
            continue
    raise ValueError(
        f"destination path is not within an allowed output root: {path_str}"
    )


def _atomic_write_path(dest: Path, *, source_path: Path | None = None, content: bytes | None = None, overwrite: bool) -> int:
    if dest.exists() and not overwrite:
        raise FileExistsError(f"destination already exists and overwrite=false: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        if source_path is not None:
            shutil.copyfile(source_path, tmp)
        else:
            tmp.write_bytes(content or b"")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dest.stat().st_size


def _stage_artifact(job_id: str, name: str, *, source_path: Path | None = None, content: bytes | None = None, suffix: str) -> tuple[Path, int]:
    job_dir = _STAGING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    staged = job_dir / f"{name}{suffix}"
    if source_path is not None:
        shutil.copyfile(source_path, staged)
    else:
        staged.write_bytes(content or b"")
    return staged, staged.stat().st_size


def _write_artifact(job_id: str, name: str, *, source_path: Path | None, content: bytes | None, destination: str | None, overwrite: bool, suffix: str) -> dict[str, Any]:
    if destination:
        dest = _resolve_destination(destination)
        size = _atomic_write_path(dest, source_path=source_path, content=content, overwrite=overwrite)
        return {"written": True, "path": str(dest), "bytes": size, "staged": False}

    staged_path, size = _stage_artifact(job_id, name, source_path=source_path, content=content, suffix=suffix)
    with _JOBS_LOCK:
        _JOBS[job_id]["staged_files"][name] = staged_path
        _JOBS[job_id]["staged_expires_at"] = time.time() + _DEFAULT_STAGING_TTL_MIN * 60
    return {
        "written": True,
        "staged": True,
        "bytes": size,
        "download_path": f"/jobs/{job_id}/artifacts/{name}",
        "expires_in_min": _DEFAULT_STAGING_TTL_MIN,
    }


def _run_job(job_id: str, source_path: Path, options: dict[str, Any], artifacts_requested: dict[str, dict[str, Any]]) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "processing"

    try:
        preflight = inspect_pdf(source_path)
        decision = preflight["decision"]

        with _JOBS_LOCK:
            _JOBS[job_id]["preflight"] = preflight

        if decision == "block":
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "blocked"
            return

        if decision == "process":
            result = run_ocr_file(
                source_path,
                mode=options.get("mode"),
                lang=options.get("lang"),
                deskew=options.get("deskew"),
                clean=options.get("clean"),
                remove_vectors=options.get("remove_vectors"),
                psm=options.get("psm"),
                jobs=options.get("jobs"),
                mask_stamps=options.get("mask_stamps"),
                mask_signatures=options.get("mask_signatures"),
                mask_grayscale=options.get("mask_grayscale"),
                mask_dilate=options.get("mask_dilate"),
                stamp_min_area=options.get("stamp_min_area"),
                stamp_max_area=options.get("stamp_max_area"),
                stamp_circularity=options.get("stamp_circularity"),
                stamp_rect_aspect_min=options.get("stamp_rect_aspect_min"),
                stamp_rect_aspect_max=options.get("stamp_rect_aspect_max"),
                signature_region=options.get("signature_region"),
            )
            base_pdf_path = Path(result["output_pdf"])
            text_source_kind = "ocr_pdf"
        else:
            base_pdf_path = source_path
            text_source_kind = "source_pdf"

        artifacts_result: dict[str, Any] = {}

        if "pdf" in artifacts_requested:
            spec = artifacts_requested["pdf"]
            artifacts_result["pdf"] = _write_artifact(
                job_id,
                "pdf",
                source_path=base_pdf_path,
                content=None,
                destination=spec.get("path"),
                overwrite=bool(spec.get("overwrite", False)),
                suffix=".pdf",
            )

        if "text" in artifacts_requested:
            spec = artifacts_requested["text"]
            content, stats = build_text_document(
                base_pdf_path,
                text_source_kind=text_source_kind,
                metadata=options.get("metadata"),
            )
            artifacts_result["text"] = _write_artifact(
                job_id,
                "text",
                source_path=None,
                content=content,
                destination=spec.get("path"),
                overwrite=bool(spec.get("overwrite", False)),
                suffix=".jsonl",
            )
            artifacts_result["text"]["stats"] = stats

        with _JOBS_LOCK:
            _JOBS[job_id]["artifacts"] = artifacts_result
            _JOBS[job_id]["status"] = "done"

    except Exception as exc:  # noqa: BLE001 - se reporta en el status del job
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(exc)


def submit_job(source_path: Path, options: dict[str, Any], artifacts_requested: dict[str, dict[str, Any]]) -> str:
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "artifacts": {},
            "staged_files": {},
            "staged_expires_at": None,
            "preflight": None,
            "error": None,
        }
    _EXECUTOR.submit(_run_job, job_id, source_path, options, artifacts_requested)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


def get_staged_artifact_path(job_id: str, name: str) -> Path | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        expires_at = job.get("staged_expires_at")
        if expires_at is not None and time.time() > expires_at:
            return None
        return job.get("staged_files", {}).get(name)


def _sweep_expired_staging() -> None:
    now = time.time()
    with _JOBS_LOCK:
        expired = [
            job_id
            for job_id, job in _JOBS.items()
            if job.get("staged_expires_at") is not None and now > job["staged_expires_at"]
        ]
    for job_id in expired:
        job_dir = _STAGING_DIR / job_id
        shutil.rmtree(job_dir, ignore_errors=True)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["staged_files"] = {}


def _staging_sweeper_loop() -> None:
    interval = max(30, _DEFAULT_STAGING_TTL_MIN * 60 // 4)
    while True:
        time.sleep(interval)
        try:
            _sweep_expired_staging()
        except Exception:
            pass


def start_staging_sweeper() -> None:
    thread = threading.Thread(target=_staging_sweeper_loop, daemon=True, name="ocr-job-staging-sweeper")
    thread.start()

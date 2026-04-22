"""
OCR Worker — consumidor activo de la cola de Normatividad en Munis.

Flujo por iteración:
  1. pull_next()           → pide el siguiente trabajo a Munis
  2. 204 → esperar y reintentar
  3. Trabajo disponible:
       a. mark_processing()  → notifica inicio
       b. download_source()  → descarga PDF original
       c. preflight()        → clasifica si requiere OCR
       d. run_ocr_file()     → aplica OCR solo cuando aporta valor real
       e. complete()         → sube PDF resultado
  4. Cualquier error       → fail() + esperar

Variables de entorno relevantes:
  MUNIS_BASE_URL                    URL base de Munis (ej. http://munis:8000)
  MUNIS_OCR_TOKEN                   Bearer token para autenticar contra Munis
  OCR_WORKER_NAME                   Nombre descriptivo del worker (logs)
  OCR_WORKER_CONCURRENCY            Cantidad de consumidores paralelos dentro del servicio (default: 1)
  OCR_POLL_INTERVAL_SECONDS         Segundos a esperar si la cola está vacía (default: 10)
  OCR_WORKER_ENABLED                "false" para deshabilitar sin borrar el contenedor (default: true)
  OCR_CALLBACK_TIMEOUT_SECONDS      Timeout HTTP para callbacks a Munis (default: 30)
  OCR_DOWNLOAD_TIMEOUT_SECONDS      Timeout HTTP para descarga del PDF fuente (default: 120)
  OCR_WORKER_MAX_CONSECUTIVE_ERRORS Detener el worker tras N errores consecutivos (default: 20)

  Además acepta todas las variables OCR de ocr_pipeline.py:
  OCR_TMP_DIR, OCR_OUT_DIR, OCR_MODE, OCR_LANG, OCR_MASK_STAMPS, etc.
"""

import logging
import multiprocessing as mp
import os
import signal
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from app import munis_client
from app.ocr_pipeline import _extract_text, run_ocr_file
from app.preflight import inspect_pdf

# ---------------------------------------------------------------------------
# Configuración del worker
# ---------------------------------------------------------------------------

WORKER_NAME: str = os.getenv("OCR_WORKER_NAME", "ocr-worker")
WORKER_CONCURRENCY: int = max(1, int(os.getenv("OCR_WORKER_CONCURRENCY", "1")))
POLL_INTERVAL: float = float(os.getenv("OCR_POLL_INTERVAL_SECONDS", "10"))
WORKER_ENABLED: bool = os.getenv("OCR_WORKER_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
)
MAX_CONSECUTIVE_ERRORS: int = int(os.getenv("OCR_WORKER_MAX_CONSECUTIVE_ERRORS", "20"))

TMP_DIR = Path(os.getenv("OCR_TMP_DIR", "/data/tmp"))
OUT_DIR = Path(os.getenv("OCR_OUT_DIR", "/data/out"))
SHARED_CACHE_DIR = Path(os.getenv("OCR_SHARED_CACHE_DIR", "/nfs-cache"))
SUPPORTED_ARTIFACTS = {"pdf": True, "txt": True, "md": False}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("OCR_WORKER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(WORKER_NAME)

# ---------------------------------------------------------------------------
# Señales de parada limpia
# ---------------------------------------------------------------------------

_stop_requested = False


def _handle_signal(signum, _frame):
    global _stop_requested
    logger.info("Señal %s recibida, deteniendo worker...", signum)
    _stop_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _worker_name_for_index(index: int) -> str:
    if WORKER_CONCURRENCY <= 1:
        return WORKER_NAME
    return f"{WORKER_NAME}-{index + 1}"


def _worker_logger(worker_name: str) -> logging.Logger:
    return logging.getLogger(worker_name)


# ---------------------------------------------------------------------------
# Procesamiento de un item de la cola
# ---------------------------------------------------------------------------


def _process_item(item: dict, worker_name: str, worker_logger: logging.Logger) -> None:
    """
    Procesa un item de la cola Munis de principio a fin.

    Args:
        item: Diccionario retornado por pull_next(). Debe contener queue_id.
    """
    payload = item.get("data", item)
    queue_id = payload.get("queue_id") or payload.get("id")
    if not queue_id:
        raise ValueError(f"Item sin queue_id: {item}")

    worker_logger.info(
        "Iniciando procesamiento queue_id=%s | pull_next_meta=%s",
        queue_id,
        _safe_meta(payload),
    )

    processing_response = munis_client.mark_processing(queue_id, worker_name=worker_name)
    processing_payload = processing_response.get("data", processing_response)

    worker_logger.info(
        "queue_id=%s marcado processing | processing_meta=%s",
        queue_id,
        _safe_meta(processing_payload) if isinstance(processing_payload, dict) else processing_payload,
    )

    requested_artifacts = _requested_artifacts(payload)
    expected_artifacts = _expected_artifacts(payload)
    unsupported_artifacts = {
        name: True
        for name, requested in requested_artifacts.items()
        if requested and not SUPPORTED_ARTIFACTS.get(name, False)
    }
    planned_artifacts = {
        name: True
        for name, requested in requested_artifacts.items()
        if requested and SUPPORTED_ARTIFACTS.get(name, False)
    }

    if not planned_artifacts:
        worker_logger.info(
            "queue_id=%s omitido: la cola no solicita artefactos soportados | requested=%s",
            queue_id,
            requested_artifacts,
        )
        munis_client.skip(
            queue_id,
            message="Cola omitida: este worker solo atiende artefactos PDF/TXT",
            reason_code="no_supported_artifacts_requested",
            duration_ms=0,
            preflight={
                "decision": "skip",
                "reason_code": "no_supported_artifacts_requested",
                "requested_artifacts": requested_artifacts,
                "unsupported_artifacts": unsupported_artifacts,
                "expected_artifacts": expected_artifacts,
            },
        )
        return

    start_time = time.monotonic()
    token = uuid.uuid4().hex[:12]
    src_pdf = TMP_DIR / f"munis_{queue_id}_{token}.pdf"
    generated_paths: list[Path] = []
    produced_artifacts: dict[str, bool] = {}
    reportable_artifacts: dict[str, bool] = {}
    shared_pdf_path = _resolve_expected_artifact_path(payload, "pdf")
    shared_txt_path = _resolve_expected_artifact_path(payload, "txt")
    pdf_cached = _artifact_cached(payload, "pdf", shared_pdf_path)
    txt_cached = _artifact_cached(payload, "txt", shared_txt_path)
    base_pdf_path: Path | None = shared_pdf_path if pdf_cached else None
    result: dict | None = None
    text_source_kind: str | None = None
    preflight: dict | None = None

    try:
        needs_base_pdf = bool(planned_artifacts.get("pdf") or planned_artifacts.get("txt"))
        if needs_base_pdf and not pdf_cached:
            munis_client.download_source(queue_id, src_pdf)
            worker_logger.info("PDF descargado: %s (%d bytes)", src_pdf, src_pdf.stat().st_size)

            preflight = inspect_pdf(src_pdf)
            worker_logger.info(
                "Preflight queue_id=%s | class=%s decision=%s signed=%s useful_text=%s",
                queue_id,
                preflight.get("classification"),
                preflight.get("decision"),
                preflight.get("signed"),
                preflight.get("has_useful_text"),
            )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            if preflight["decision"] == "skip":
                if planned_artifacts.get("txt"):
                    base_pdf_path = src_pdf
                    text_source_kind = "source_pdf"
                    worker_logger.info(
                        "queue_id=%s omite OCR por texto útil y usará el PDF fuente para artefactos textuales",
                        queue_id,
                    )
                else:
                    munis_client.skip(
                        queue_id,
                        message=preflight["message"],
                        reason_code=preflight.get("reason_code"),
                        duration_ms=elapsed_ms,
                        preflight=preflight,
                    )
                    worker_logger.info(
                        "queue_id=%s omitido por preflight | reason=%s",
                        queue_id,
                        preflight.get("reason_code"),
                    )
                    return

            if preflight["decision"] == "block":
                munis_client.block(
                    queue_id,
                    message=preflight["message"],
                    reason_code=preflight.get("reason_code"),
                    duration_ms=elapsed_ms,
                    preflight=preflight,
                )
                worker_logger.info(
                    "queue_id=%s bloqueado por preflight | reason=%s",
                    queue_id,
                    preflight.get("reason_code"),
                )
                return

            if preflight["decision"] != "skip":
                try:
                    result = run_ocr_file(src_pdf, tmp_dir=TMP_DIR, out_dir=OUT_DIR)
                except Exception as exc:
                    if _looks_like_signature_error(exc):
                        elapsed_ms = int((time.monotonic() - start_time) * 1000)
                        fallback_preflight = {
                            **preflight,
                            "decision": "block",
                            "reason_code": "digital_signature_detected_during_ocr",
                            "message": "OCR bloqueado por firma digital detectada durante el procesamiento",
                        }
                        munis_client.block(
                            queue_id,
                            message=fallback_preflight["message"],
                            reason_code=fallback_preflight["reason_code"],
                            duration_ms=elapsed_ms,
                            preflight=fallback_preflight,
                        )
                        worker_logger.warning(
                            "queue_id=%s bloqueado por firma detectada durante OCR",
                            queue_id,
                        )
                        return
                    raise

                generated_paths = _result_artifact_paths(result)

                out_pdf = result.get("output_pdf")
                if not out_pdf:
                    raise RuntimeError(f"run_ocr_file no devolvió output_pdf. result={result}")

                out_pdf_path = Path(out_pdf)
                if not out_pdf_path.exists():
                    raise RuntimeError(f"output_pdf no existe en disco: {out_pdf_path}")

                if shared_pdf_path is not None:
                    _publish_pdf_to_shared_cache(out_pdf_path, shared_pdf_path)
                    base_pdf_path = shared_pdf_path
                else:
                    base_pdf_path = out_pdf_path
                pdf_cached = True
                produced_artifacts["pdf"] = True
                if text_source_kind is None:
                    text_source_kind = "ocr_pdf"
                worker_logger.info(
                    "OCR PDF base listo queue_id=%s | pdf=%s",
                    queue_id,
                    base_pdf_path,
                )

        if planned_artifacts.get("txt"):
            if txt_cached and shared_txt_path is not None:
                if text_source_kind is None:
                    text_source_kind = "shared_cache_txt"
                worker_logger.info(
                    "TXT reutilizado desde cache queue_id=%s | txt=%s",
                    queue_id,
                    shared_txt_path,
                )
            else:
                if base_pdf_path is None or not base_pdf_path.exists():
                    raise RuntimeError(
                        "No hay PDF base disponible para derivar TXT del cache OCR"
                    )
                if shared_txt_path is None:
                    raise RuntimeError(
                        "La cola solicita TXT pero artifacts.expected.txt_path no fue provisto"
                    )
                txt_content = _generate_txt_from_pdf(base_pdf_path)
                _publish_text_to_shared_cache(txt_content, shared_txt_path)
                txt_cached = True
                produced_artifacts["txt"] = True
                if text_source_kind is None:
                    text_source_kind = "source_pdf" if base_pdf_path == src_pdf else "ocr_pdf"
                worker_logger.info(
                    "TXT generado queue_id=%s | txt=%s | text_len=%d",
                    queue_id,
                    shared_txt_path,
                    len(txt_content),
                )

        if requested_artifacts.get("pdf") and pdf_cached:
            reportable_artifacts["pdf"] = True
        elif produced_artifacts.get("pdf"):
            reportable_artifacts["pdf"] = True

        if requested_artifacts.get("txt") and txt_cached:
            reportable_artifacts["txt"] = True

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        response_payload = {
            "requested_artifacts": requested_artifacts,
            "planned_artifacts": planned_artifacts,
            "unsupported_artifacts": unsupported_artifacts,
            "produced_artifacts": produced_artifacts,
            "reported_artifacts": reportable_artifacts,
            "expected_artifacts": expected_artifacts,
            "storage": {
                "shared_cache_dir": str(SHARED_CACHE_DIR),
                "resolved_pdf_path": str(base_pdf_path) if base_pdf_path else None,
                "resolved_txt_path": str(shared_txt_path) if shared_txt_path else None,
            },
            "preflight": preflight,
            "text_source_kind": text_source_kind,
            "ocr": {
                "mode": result.get("mode") if result else None,
                "text_len": result.get("text_len") if result else None,
                "elapsed_sec": result.get("elapsed_sec") if result else None,
            },
        }
        engine_response = {
            "result": {
                "mode": result.get("mode") if result else None,
                "output_pdf": str(base_pdf_path) if base_pdf_path else None,
                "output_txt": str(shared_txt_path) if txt_cached and shared_txt_path else None,
                "text_len": result.get("text_len") if result else None,
                "elapsed_sec": result.get("elapsed_sec") if result else None,
            },
            "source": payload.get("source") or {},
            "document": payload.get("document") or {},
            "artifacts": payload.get("artifacts") or {},
            "ocr_flags": _ocr_artifact_flags(payload),
            "preflight": preflight,
            "text_source_kind": text_source_kind,
        }

        if expected_artifacts:
            if reportable_artifacts.get("pdf") and shared_pdf_path is None:
                raise RuntimeError(
                    "La cola reporta pdf pero artifacts.expected.pdf_path no fue provisto"
                )
            if reportable_artifacts.get("txt") and shared_txt_path is None:
                raise RuntimeError(
                    "La cola reporta txt pero artifacts.expected.txt_path no fue provisto"
                )
            munis_client.report_artifacts(
                queue_id,
                processor=worker_name,
                duration_ms=elapsed_ms,
                finalize_queue=_should_finalize_queue(
                    planned_artifacts,
                    reportable_artifacts,
                    unsupported_artifacts,
                ),
                artifacts=reportable_artifacts,
                response_payload_json=response_payload,
                engine_response_json=engine_response,
            )
            worker_logger.info(
                "Item queue_id=%s reportado por /artifacts | artifacts=%s",
                queue_id,
                reportable_artifacts,
            )
        elif produced_artifacts.get("pdf") and base_pdf_path is not None:
            munis_client.complete(queue_id, base_pdf_path, duration_ms=elapsed_ms)
            worker_logger.info(
                "Item queue_id=%s completado por flujo legacy multipart",
                queue_id,
            )
        else:
            raise RuntimeError(
                "No hay artifacts.expected para reportar y no se produjo un PDF para fallback legacy"
            )
        _cleanup(*generated_paths)

    except munis_client.QueueSourceUnavailableError as exc:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        worker_logger.warning(
            "queue_id=%s sin archivo fuente disponible: %s",
            queue_id,
            exc.message,
        )
        munis_client.fail(queue_id, exc.message, duration_ms=elapsed_ms)

    finally:
        _cleanup(src_pdf)


def _safe_meta(item: dict) -> str:
    meta = {k: item[k] for k in ("queue_id", "id", "status", "regulation_file_id", "lease_expires_at") if k in item}
    artifacts = item.get("artifacts")
    if isinstance(artifacts, dict):
        requested = artifacts.get("requested")
        expected = artifacts.get("expected")
        if requested is not None:
            meta["requested_artifacts"] = requested
        if isinstance(expected, dict) and "dir" in expected:
            meta["artifacts_dir"] = expected["dir"]
    return str(meta)


def _requested_artifacts(payload: dict) -> dict[str, bool]:
    artifacts = payload.get("artifacts")
    requested = artifacts.get("requested") if isinstance(artifacts, dict) else None
    if not isinstance(requested, dict):
        return {"pdf": True, "txt": False, "md": False}
    return {
        "pdf": bool(requested.get("pdf")),
        "txt": bool(requested.get("txt")),
        "md": bool(requested.get("md")),
    }


def _expected_artifacts(payload: dict) -> dict:
    artifacts = payload.get("artifacts")
    expected = artifacts.get("expected") if isinstance(artifacts, dict) else None
    return expected if isinstance(expected, dict) else {}


def _ocr_artifact_flags(payload: dict) -> dict[str, bool | None]:
    candidates = []
    for key in ("ocr", "ocr_record", "file_ocr"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(payload)
    for candidate in candidates:
        flags = {}
        found = False
        for name in ("pdf", "txt", "md"):
            key = f"has_{name}"
            if key in candidate:
                flags[name] = bool(candidate.get(key))
                found = True
            else:
                flags[name] = None
        if found:
            return flags
    return {"pdf": None, "txt": None, "md": None}


def _artifact_cached(payload: dict, artifact: str, resolved_path: Path | None) -> bool:
    flags = _ocr_artifact_flags(payload)
    has_flag = flags.get(artifact)
    if resolved_path is None:
        return bool(has_flag)
    if has_flag is False:
        return False
    return resolved_path.exists()


def _resolve_expected_artifact_path(payload: dict, artifact: str) -> Path | None:
    expected = _expected_artifacts(payload)
    raw = expected.get(f"{artifact}_path")
    if not raw:
        return None
    relative = Path(str(raw))
    if relative.is_absolute():
        raise RuntimeError(
            f"expected {artifact}_path debe ser relativo al cache compartido: {raw}"
        )
    candidate = (SHARED_CACHE_DIR / relative).resolve()
    shared_root = SHARED_CACHE_DIR.resolve()
    try:
        candidate.relative_to(shared_root)
    except ValueError as exc:
        raise RuntimeError(
            f"expected {artifact}_path fuera del cache compartido: {raw}"
        ) from exc
    return candidate


def _publish_pdf_to_shared_cache(src_pdf: Path, target_pdf: Path) -> None:
    target_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_pdf.with_suffix(f"{target_pdf.suffix}.tmp")
    shutil.copy2(src_pdf, tmp_target)
    os.replace(tmp_target, target_pdf)


def _publish_text_to_shared_cache(content: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_path.with_suffix(f"{target_path.suffix}.tmp")
    tmp_target.write_text(content, encoding="utf-8")
    os.replace(tmp_target, target_path)


def _generate_txt_from_pdf(pdf_path: Path) -> str:
    content = _extract_text(pdf_path)
    if not content.strip():
        raise RuntimeError(f"TXT derivado vacío desde PDF base: {pdf_path}")
    return content


def _should_finalize_queue(
    planned_artifacts: dict[str, bool],
    reportable_artifacts: dict[str, bool],
    unsupported_artifacts: dict[str, bool],
) -> bool:
    if unsupported_artifacts:
        return False
    for artifact, planned in planned_artifacts.items():
        if planned and not reportable_artifacts.get(artifact, False):
            return False
    return True


def _looks_like_signature_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return "digitalsignatureerror" in text or "digital signature" in text


def _result_artifact_paths(result: dict) -> list[Path]:
    artifact_keys = (
        "output_pdf",
        "masked_pdf",
        "masked_output_pdf",
        "original_searchable_pdf",
        "working_searchable_pdf",
        "detected_pdf",
        "paddle_masked_searchable_pdf",
        "overlap_debug_pdf",
        "overlap_debug_searchable_pdf",
        "overlap_debug_working_searchable_pdf",
        "paddle_searchable_pdf",
        "paddle_json",
        "paddle_text",
    )
    paths: list[Path] = []
    for key in artifact_keys:
        raw = result.get(key)
        if not raw:
            continue
        path = Path(raw)
        if path not in paths:
            paths.append(path)
    return paths


def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            if p and p.exists():
                p.unlink()
                logger.debug("Limpiado: %s", p)
        except Exception as exc:
            logger.warning("No se pudo limpiar %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


def _run_worker_loop(worker_name: str) -> None:
    worker_logger = _worker_logger(worker_name)

    if not WORKER_ENABLED:
        worker_logger.info("OCR_WORKER_ENABLED=false → worker deshabilitado, saliendo.")
        return

    if not munis_client.MUNIS_BASE_URL:
        worker_logger.critical(
            "MUNIS_BASE_URL no está configurado. El worker no puede iniciar."
        )
        sys.exit(1)

    if not munis_client.MUNIS_OCR_TOKEN:
        worker_logger.warning(
            "MUNIS_OCR_TOKEN está vacío. Las peticiones a Munis probablemente fallarán."
        )

    worker_logger.info(
        "Worker '%s' iniciado | MUNIS_BASE_URL=%s | poll_interval=%.1fs",
        worker_name,
        munis_client.MUNIS_BASE_URL,
        POLL_INTERVAL,
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    consecutive_errors = 0

    while not _stop_requested:
        try:
            item = munis_client.pull_next(worker_name=worker_name)

            if item is None:
                worker_logger.debug("Cola vacía, esperando %.1fs...", POLL_INTERVAL)
                _interruptible_sleep(POLL_INTERVAL)
                consecutive_errors = 0
                continue

            consecutive_errors = 0
            payload = item.get("data", item)
            queue_id = payload.get("queue_id") or payload.get("id")

            item_start = time.monotonic()
            try:
                _process_item(item, worker_name, worker_logger)
            except Exception as exc:
                worker_logger.exception("Error procesando queue_id=%s: %s", queue_id, exc)
                elapsed_ms = int((time.monotonic() - item_start) * 1000)
                try:
                    munis_client.fail(queue_id, str(exc), duration_ms=elapsed_ms)
                except Exception as fail_exc:
                    worker_logger.error(
                        "No se pudo reportar fallo a Munis (queue_id=%s): %s",
                        queue_id,
                        fail_exc,
                    )
                _interruptible_sleep(POLL_INTERVAL)

        except Exception as poll_exc:
            consecutive_errors += 1
            worker_logger.error(
                "Error en pull_next (error %d/%d): %s",
                consecutive_errors,
                MAX_CONSECUTIVE_ERRORS,
                poll_exc,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                worker_logger.critical(
                    "Se alcanzó el límite de %d errores consecutivos. Deteniendo worker.",
                    MAX_CONSECUTIVE_ERRORS,
                )
                sys.exit(1)

            wait = min(POLL_INTERVAL * (2 ** min(consecutive_errors - 1, 4)), POLL_INTERVAL * 5)
            worker_logger.info("Esperando %.1fs antes de reintentar...", wait)
            _interruptible_sleep(wait)

    worker_logger.info("Worker '%s' detenido limpiamente.", worker_name)


def process_single_item(
    item: dict[str, Any],
    worker_name: str | None = None,
    *,
    raise_errors: bool = True,
) -> None:
    """
    Procesa un item ya obtenido previamente, sin hacer polling continuo.
    Útil para depuración uno por uno.
    """
    effective_worker_name = (worker_name or WORKER_NAME).strip() or WORKER_NAME
    worker_logger = _worker_logger(effective_worker_name)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = item.get("data", item)
    queue_id = payload.get("queue_id") or payload.get("id")
    item_start = time.monotonic()

    try:
        _process_item(item, effective_worker_name, worker_logger)
    except Exception as exc:
        worker_logger.exception("Error procesando queue_id=%s: %s", queue_id, exc)
        elapsed_ms = int((time.monotonic() - item_start) * 1000)
        try:
            munis_client.fail(queue_id, str(exc), duration_ms=elapsed_ms)
        except Exception as fail_exc:
            worker_logger.error(
                "No se pudo reportar fallo a Munis (queue_id=%s): %s",
                queue_id,
                fail_exc,
            )
        if raise_errors:
            raise


def run_worker_once(worker_name: str | None = None) -> bool:
    """
    Hace un solo pull-next y procesa como máximo un item, luego sale.

    Returns:
        True si encontró y procesó un item, False si la cola estaba vacía.
    """
    effective_worker_name = (worker_name or WORKER_NAME).strip() or WORKER_NAME
    worker_logger = _worker_logger(effective_worker_name)

    if not WORKER_ENABLED:
        worker_logger.info("OCR_WORKER_ENABLED=false → worker deshabilitado, saliendo.")
        return False

    if not munis_client.MUNIS_BASE_URL:
        worker_logger.critical(
            "MUNIS_BASE_URL no está configurado. El worker no puede iniciar."
        )
        sys.exit(1)

    if not munis_client.MUNIS_OCR_TOKEN:
        worker_logger.warning(
            "MUNIS_OCR_TOKEN está vacío. Las peticiones a Munis probablemente fallarán."
        )

    worker_logger.info(
        "Worker once '%s' iniciado | MUNIS_BASE_URL=%s",
        effective_worker_name,
        munis_client.MUNIS_BASE_URL,
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    item = munis_client.pull_next(worker_name=effective_worker_name)
    if item is None:
        worker_logger.info("Cola vacía: no se encontró item para procesar.")
        return False

    process_single_item(item, effective_worker_name)
    worker_logger.info("Worker once '%s' terminó tras procesar un item.", effective_worker_name)
    return True


def _run_worker_process(index: int) -> None:
    global _stop_requested
    _stop_requested = False
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _run_worker_loop(_worker_name_for_index(index))


def _terminate_children(children: list[mp.Process]) -> None:
    for proc in children:
        if proc.is_alive():
            proc.terminate()


def run_worker() -> None:
    if WORKER_CONCURRENCY <= 1:
        _run_worker_loop(WORKER_NAME)
        return

    logger.info(
        "Supervisor iniciando %d consumidores para la misma cola OCR",
        WORKER_CONCURRENCY,
    )
    children: list[mp.Process] = []
    for index in range(WORKER_CONCURRENCY):
        proc = mp.Process(
            target=_run_worker_process,
            args=(index,),
            name=_worker_name_for_index(index),
        )
        proc.start()
        children.append(proc)
        logger.info("Consumidor %s iniciado con pid=%s", proc.name, proc.pid)

    exit_code = 0
    try:
        while children:
            for proc in list(children):
                proc.join(timeout=0.5)
                if proc.is_alive():
                    continue
                children.remove(proc)
                if _stop_requested:
                    continue
                logger.error(
                    "Consumidor %s terminó inesperadamente con exitcode=%s",
                    proc.name,
                    proc.exitcode,
                )
                exit_code = proc.exitcode or 1
                _terminate_children(children)
                children = []
                break
            if _stop_requested:
                _terminate_children(children)
                break
    finally:
        for proc in children:
            proc.join(timeout=5)

    if _stop_requested:
        logger.info("Supervisor detenido limpiamente.")
        return
    if exit_code:
        sys.exit(exit_code)


def _interruptible_sleep(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not _stop_requested and time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))

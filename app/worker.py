"""
OCR Worker — consumidor activo de la cola de Normatividad en Munis.

Flujo por iteración:
  1. pull_next()           → pide el siguiente trabajo a Munis
  2. 204 → esperar y reintentar
  3. Trabajo disponible:
       a. mark_processing()  → notifica inicio
       b. download_source()  → descarga PDF original
       c. run_ocr_file()     → aplica OCR (lógica existente)
       d. complete()         → sube PDF resultado
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
import sys
import time
import uuid
from pathlib import Path

from app import munis_client
from app.ocr_pipeline import run_ocr_file

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

    worker_logger.info("Iniciando procesamiento queue_id=%s | meta=%s", queue_id, _safe_meta(payload))

    munis_client.mark_processing(queue_id, worker_name=worker_name)

    token = uuid.uuid4().hex[:12]
    src_pdf = TMP_DIR / f"munis_{queue_id}_{token}.pdf"
    start_time = time.monotonic()
    generated_paths: list[Path] = []

    try:
        munis_client.download_source(queue_id, src_pdf)
        worker_logger.info("PDF descargado: %s (%d bytes)", src_pdf, src_pdf.stat().st_size)

        result = run_ocr_file(src_pdf, tmp_dir=TMP_DIR, out_dir=OUT_DIR)
        generated_paths = _result_artifact_paths(result)

        out_pdf = result.get("output_pdf")
        if not out_pdf:
            raise RuntimeError(f"run_ocr_file no devolvió output_pdf. result={result}")

        out_pdf_path = Path(out_pdf)
        if not out_pdf_path.exists():
            raise RuntimeError(f"output_pdf no existe en disco: {out_pdf_path}")

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        worker_logger.info(
            "OCR completado queue_id=%s | text_len=%s duration=%dms",
            queue_id,
            result.get("text_len", "?"),
            elapsed_ms,
        )

        munis_client.complete(queue_id, out_pdf_path, duration_ms=elapsed_ms)
        worker_logger.info("Item queue_id=%s completado exitosamente", queue_id)
        _cleanup(*generated_paths)

    finally:
        _cleanup(src_pdf)


def _safe_meta(item: dict) -> str:
    keys = ("queue_id", "id", "status", "regulation_file_id", "lease_expires_at")
    return str({k: item[k] for k in keys if k in item})


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

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
  MUNIS_BASE_URL                  URL base de Munis (ej. http://munis:8000)
  MUNIS_OCR_TOKEN                 Bearer token para autenticar contra Munis
  OCR_WORKER_NAME                 Nombre descriptivo del worker (logs)
  OCR_POLL_INTERVAL_SECONDS       Segundos a esperar si la cola está vacía (default: 10)
  OCR_WORKER_ENABLED              "false" para deshabilitar sin borrar el contenedor (default: true)
  OCR_CALLBACK_TIMEOUT_SECONDS    Timeout HTTP para callbacks a Munis (default: 30)
  OCR_DOWNLOAD_TIMEOUT_SECONDS    Timeout HTTP para descarga del PDF fuente (default: 120)
  OCR_WORKER_MAX_CONSECUTIVE_ERRORS Detener el worker tras N errores consecutivos (default: 20)

  Además acepta todas las variables OCR de ocr_pipeline.py:
  OCR_TMP_DIR, OCR_OUT_DIR, OCR_MODE, OCR_LANG, OCR_MASK_STAMPS, etc.
"""

import logging
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


# ---------------------------------------------------------------------------
# Procesamiento de un item de la cola
# ---------------------------------------------------------------------------


def _process_item(item: dict) -> None:
    """
    Procesa un item de la cola Munis de principio a fin.

    Args:
        item: Diccionario retornado por pull_next(). Debe contener queue_id.
    """
    # Munis envuelve la respuesta en {"data": {...}}
    payload = item.get("data", item)
    queue_id = payload.get("queue_id") or payload.get("id")
    if not queue_id:
        raise ValueError(f"Item sin queue_id: {item}")

    logger.info("Iniciando procesamiento queue_id=%s | meta=%s", queue_id, _safe_meta(payload))

    # 1. Marcar como "en proceso"
    munis_client.mark_processing(queue_id)

    # 2. Descargar PDF fuente
    token = uuid.uuid4().hex[:12]
    src_pdf = TMP_DIR / f"munis_{queue_id}_{token}.pdf"
    start_time = time.monotonic()
    generated_paths: list[Path] = []

    try:
        munis_client.download_source(queue_id, src_pdf)
        logger.info("PDF descargado: %s (%d bytes)", src_pdf, src_pdf.stat().st_size)

        # 3. Aplicar OCR
        result = run_ocr_file(src_pdf, tmp_dir=TMP_DIR, out_dir=OUT_DIR)
        generated_paths = _result_artifact_paths(result)

        out_pdf = result.get("output_pdf")
        if not out_pdf:
            raise RuntimeError(f"run_ocr_file no devolvió output_pdf. result={result}")

        out_pdf_path = Path(out_pdf)
        if not out_pdf_path.exists():
            raise RuntimeError(f"output_pdf no existe en disco: {out_pdf_path}")

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "OCR completado queue_id=%s | text_len=%s duration=%dms",
            queue_id,
            result.get("text_len", "?"),
            elapsed_ms,
        )

        # 4. Enviar resultado a Munis
        munis_client.complete(queue_id, out_pdf_path, duration_ms=elapsed_ms)
        logger.info("Item queue_id=%s completado exitosamente", queue_id)
        _cleanup(*generated_paths)

    except Exception:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        raise
    finally:
        # Limpiar archivos temporales del worker
        _cleanup(src_pdf)


def _safe_meta(item: dict) -> str:
    """
    Extrae campos útiles del item para logging.
    El campo `status` ya viene normalizado a string desde munis_client.normalize_payload.
    """
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
    """Elimina archivos temporales silenciosamente."""
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


def run_worker() -> None:
    """
    Ejecuta el loop de polling continuo.

    Sale limpiamente con SystemExit(0) si se recibe SIGTERM/SIGINT
    o si se supera MAX_CONSECUTIVE_ERRORS errores seguidos.
    """
    if not WORKER_ENABLED:
        logger.info("OCR_WORKER_ENABLED=false → worker deshabilitado, saliendo.")
        return

    if not munis_client.MUNIS_BASE_URL:
        logger.critical(
            "MUNIS_BASE_URL no está configurado. El worker no puede iniciar."
        )
        sys.exit(1)

    if not munis_client.MUNIS_OCR_TOKEN:
        logger.warning(
            "MUNIS_OCR_TOKEN está vacío. Las peticiones a Munis probablemente fallarán."
        )

    logger.info(
        "Worker '%s' iniciado | MUNIS_BASE_URL=%s | poll_interval=%.1fs",
        WORKER_NAME,
        munis_client.MUNIS_BASE_URL,
        POLL_INTERVAL,
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    consecutive_errors = 0

    while not _stop_requested:
        try:
            item = munis_client.pull_next()

            if item is None:
                # Cola vacía — esperar antes de reintentar
                logger.debug("Cola vacía, esperando %.1fs...", POLL_INTERVAL)
                _interruptible_sleep(POLL_INTERVAL)
                consecutive_errors = 0  # cola vacía no es un error
                continue

            # Tenemos trabajo
            consecutive_errors = 0
            payload = item.get("data", item)
            queue_id = payload.get("queue_id") or payload.get("id")

            item_start = time.monotonic()
            try:
                _process_item(item)
            except Exception as exc:
                logger.exception(
                    "Error procesando queue_id=%s: %s", queue_id, exc
                )
                elapsed_ms = int((time.monotonic() - item_start) * 1000)
                try:
                    munis_client.fail(queue_id, str(exc), duration_ms=elapsed_ms)
                except Exception as fail_exc:
                    logger.error(
                        "No se pudo reportar fallo a Munis (queue_id=%s): %s",
                        queue_id,
                        fail_exc,
                    )
                # Tras un fallo de procesamiento, seguir con el siguiente item
                _interruptible_sleep(POLL_INTERVAL)

        except Exception as poll_exc:
            consecutive_errors += 1
            logger.error(
                "Error en pull_next (error %d/%d): %s",
                consecutive_errors,
                MAX_CONSECUTIVE_ERRORS,
                poll_exc,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "Se alcanzó el límite de %d errores consecutivos. Deteniendo worker.",
                    MAX_CONSECUTIVE_ERRORS,
                )
                sys.exit(1)

            # Backoff exponencial suave (máx. 5× el intervalo normal)
            wait = min(POLL_INTERVAL * (2 ** min(consecutive_errors - 1, 4)), POLL_INTERVAL * 5)
            logger.info("Esperando %.1fs antes de reintentar...", wait)
            _interruptible_sleep(wait)

    logger.info("Worker '%s' detenido limpiamente.", WORKER_NAME)


def _interruptible_sleep(seconds: float) -> None:
    """Sleep que respeta la señal de parada cada segundo."""
    deadline = time.monotonic() + seconds
    while not _stop_requested and time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))

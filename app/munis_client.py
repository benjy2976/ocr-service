"""
Cliente HTTP para la API OCR de Munis (Normatividad).

Encapsula todos los calls hacia Munis: pull de cola, marcado de estado,
descarga de PDF fuente y envío del resultado OCR.

Compatibilidad de status
-------------------------
Munis puede devolver el campo `status` como string (formato legado) o como
entero smallint (formato nuevo). Este módulo normaliza ambos en un string
legible antes de exponerlos al worker o a los logs.

Catálogo regulation_file_ocr_queues.status
  0 = pending | 1 = leased | 2 = processing | 3 = done
  4 = failed  | 5 = obsolete | 6 = cancelled

Catálogo regulation_file_ocrs.status
  0 = not_requested | 1 = eligible | 2 = queued | 3 = processing
  4 = done | 5 = failed | 6 = obsolete | 7 = skipped
"""

import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MUNIS_BASE_URL: str = os.getenv("MUNIS_BASE_URL", "").rstrip("/")
MUNIS_OCR_TOKEN: str = os.getenv("MUNIS_OCR_TOKEN", "")
OCR_CALLBACK_TIMEOUT: int = int(os.getenv("OCR_CALLBACK_TIMEOUT_SECONDS", "30"))
OCR_DOWNLOAD_TIMEOUT: int = int(os.getenv("OCR_DOWNLOAD_TIMEOUT_SECONDS", "120"))
WORKER_NAME: str = os.getenv("OCR_WORKER_NAME", "ocr-worker-1")

# Activar con OCR_HTTP_DEBUG=true — solo para debug, nunca en producción
_HTTP_DEBUG: bool = os.getenv("OCR_HTTP_DEBUG", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Catálogos de estado (int → str)
# ---------------------------------------------------------------------------

# regulation_file_ocr_queues.status
QUEUE_STATUS: dict[int, str] = {
    0: "pending",
    1: "leased",
    2: "processing",
    3: "done",
    4: "failed",
    5: "obsolete",
    6: "cancelled",
}

# regulation_file_ocrs.status
FILE_OCR_STATUS: dict[int, str] = {
    0: "not_requested",
    1: "eligible",
    2: "queued",
    3: "processing",
    4: "done",
    5: "failed",
    6: "obsolete",
    7: "skipped",
}


def resolve_status(value: int | str | None, catalog: dict[int, str] = QUEUE_STATUS) -> str:
    """
    Normaliza un campo status a string legible.

    Acepta:
      - None           → "unknown"
      - str            → lo devuelve tal cual (compatibilidad legado)
      - int            → traduce con el catálogo dado; si no está, "status:<n>"

    Args:
        value:   Valor raw del campo status en la respuesta JSON.
        catalog: Diccionario int→str a usar. Por defecto QUEUE_STATUS.
    """
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return catalog.get(value, f"status:{value}")
    return str(value)


def normalize_payload(payload: dict, catalog: dict[int, str] = QUEUE_STATUS) -> dict:
    """
    Devuelve una copia del payload con el campo `status` normalizado a string.

    Prioridad:
      1. `status_label` (si Munis lo incluye, es el string oficial)
      2. `status` mapeado con el catálogo dado
      3. "unknown"

    No modifica el dict original.
    """
    result = dict(payload)
    if "status_label" in payload:
        result["status"] = str(payload["status_label"])
    else:
        result["status"] = resolve_status(payload.get("status"), catalog)
    return result


# ---------------------------------------------------------------------------
# Debug HTTP (solo cuando OCR_HTTP_DEBUG=true)
# ---------------------------------------------------------------------------


def _dbg(method: str, url: str, status: int | None = None, body: str | None = None) -> None:
    """Imprime trazas HTTP a stdout solo cuando OCR_HTTP_DEBUG=true."""
    if not _HTTP_DEBUG:
        return
    arrow = "→" if status is None else f"← {status}"
    print(f"[MUNIS HTTP] {method} {arrow} {url}", flush=True)
    if body:
        preview = body[:800].replace("\n", " ")
        print(f"             body: {preview}", flush=True)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {MUNIS_OCR_TOKEN}",
        "Accept": "application/json",
    }


def _url(path: str) -> str:
    if not MUNIS_BASE_URL:
        raise RuntimeError(
            "MUNIS_BASE_URL no está configurado. "
            "Define la variable de entorno antes de iniciar el worker."
        )
    return f"{MUNIS_BASE_URL}{path}"


# ---------------------------------------------------------------------------
# Operaciones de cola
# ---------------------------------------------------------------------------


def pull_next() -> dict | None:
    """
    POST /api/ocr/normatividad/queue/pull-next

    Solicita el siguiente trabajo pendiente.

    Returns:
        dict con los datos del item (status ya normalizado a string),
        None si la respuesta es 204 (cola vacía).
    """
    url = _url("/api/ocr/normatividad/queue/pull-next")
    _dbg("POST", url)
    headers = {**_headers(), "X-OCR-Processor": WORKER_NAME}
    resp = requests.post(url, headers=headers, timeout=OCR_CALLBACK_TIMEOUT)
    _dbg("POST", url, resp.status_code, resp.text if resp.status_code != 204 else "(vacío)")

    if resp.status_code == 204:
        return None

    resp.raise_for_status()
    body = resp.json()

    # Normalizar status dentro de data si existe
    if "data" in body and isinstance(body["data"], dict):
        body["data"] = normalize_payload(body["data"], QUEUE_STATUS)

    return body


def mark_processing(queue_id: int | str) -> dict:
    """
    POST /api/ocr/normatividad/queue/{queue_id}/processing

    Notifica a Munis que el worker tomó el item y comenzó el OCR.

    Returns:
        dict con el estado actualizado del item (status normalizado).
    """
    url = _url(f"/api/ocr/normatividad/queue/{queue_id}/processing")
    body = {"processor": WORKER_NAME}
    _dbg("POST", url, body=str(body))
    resp = requests.post(url, headers=_headers(), json=body, timeout=OCR_CALLBACK_TIMEOUT)
    _dbg("POST", url, resp.status_code, resp.text)
    resp.raise_for_status()

    data = resp.json()
    if "data" in data and isinstance(data["data"], dict):
        data["data"] = normalize_payload(data["data"], QUEUE_STATUS)
    return data


def download_source(queue_id: int | str, dest_path: Path) -> None:
    """
    GET /api/ocr/normatividad/queue/{queue_id}/source

    Descarga el PDF original y lo guarda en dest_path.
    """
    url = _url(f"/api/ocr/normatividad/queue/{queue_id}/source")
    _dbg("GET", url)
    with requests.get(url, headers=_headers(), stream=True, timeout=OCR_DOWNLOAD_TIMEOUT) as resp:
        _dbg(
            "GET", url, resp.status_code,
            f"Content-Type: {resp.headers.get('Content-Type')} | "
            f"Content-Length: {resp.headers.get('Content-Length')}",
        )
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)

    logger.debug("download_source OK: %d bytes → %s", dest_path.stat().st_size, dest_path)


def complete(queue_id: int | str, pdf_path: Path, duration_ms: int | None = None) -> dict:
    """
    POST /api/ocr/normatividad/queue/{queue_id}/complete  (multipart/form-data)

    Sube el PDF con OCR y marca el item como procesado exitosamente.

    Campos enviados:
      ocr_file    — PDF resultante (obligatorio)
      duration_ms — milisegundos de procesamiento (opcional)

    Returns:
        dict con el registro ocr actualizado (status normalizado con FILE_OCR_STATUS).
    """
    url = _url(f"/api/ocr/normatividad/queue/{queue_id}/complete")
    data: dict = {}
    if duration_ms is not None:
        data["duration_ms"] = str(duration_ms)
    _dbg("POST", url, body=f"ocr_file={pdf_path.name} | data={data}")

    with open(pdf_path, "rb") as fh:
        files = {"ocr_file": (pdf_path.name, fh, "application/pdf")}
        resp = requests.post(
            url,
            headers=_headers(),
            data=data,
            files=files,
            timeout=OCR_CALLBACK_TIMEOUT,
        )
    _dbg("POST", url, resp.status_code, resp.text)
    resp.raise_for_status()

    result = resp.json()
    if "data" in result and isinstance(result["data"], dict):
        result["data"] = normalize_payload(result["data"], FILE_OCR_STATUS)

    logger.info("complete OK queue_id=%s | status=%s", queue_id, result.get("data", {}).get("status"))
    return result


def fail(queue_id: int | str, reason: str, duration_ms: int | None = None) -> None:
    """
    POST /api/ocr/normatividad/queue/{queue_id}/fail  (JSON)

    Reporta que el procesamiento falló. Munis reencola o marca como failed
    según el número de intentos acumulados.

    Campos enviados:
      error_message — descripción del error (obligatorio)
      duration_ms   — milisegundos hasta el fallo (opcional)
    """
    url = _url(f"/api/ocr/normatividad/queue/{queue_id}/fail")
    reason_safe = str(reason)[:2000]
    body: dict = {"error_message": reason_safe}
    if duration_ms is not None:
        body["duration_ms"] = duration_ms
    _dbg("POST", url, body=str(body))

    resp = requests.post(url, headers=_headers(), json=body, timeout=OCR_CALLBACK_TIMEOUT)
    _dbg("POST", url, resp.status_code, resp.text)

    if not resp.ok:
        logger.error(
            "fail endpoint devolvió %s para queue_id=%s: %s",
            resp.status_code, queue_id, resp.text[:500],
        )
        return

    try:
        result = resp.json()
        if "data" in result and isinstance(result["data"], dict):
            data = normalize_payload(result["data"], FILE_OCR_STATUS)
            logger.info(
                "fail registrado queue_id=%s | nuevo status=%s | intentos=%s",
                queue_id, data.get("status"), data.get("attempts"),
            )
    except Exception:
        pass


def fetch_status() -> dict | None:
    """
    GET /api/ocr/normatividad/status

    Consulta el estado y configuración del servicio OCR en Munis
    (si la cola está activa, pausada, tipos permitidos, etc).

    Returns:
        dict con los settings, o None si el endpoint falla.
    """
    url = _url("/api/ocr/normatividad/status")
    _dbg("GET", url)
    try:
        resp = requests.get(url, headers=_headers(), timeout=OCR_CALLBACK_TIMEOUT)
        _dbg("GET", url, resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception as exc:
        logger.warning("No se pudo consultar /status en Munis: %s", exc)
        return None

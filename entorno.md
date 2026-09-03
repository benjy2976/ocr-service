# Variables de entorno

Inventario por repo destino (ver `arquitectura.md` para el plan de split).
Fuente normativa unica de esta lista; `README.md` puede resumir pero no
duplicar las descripciones completas.

## ocr-core (este repo, tras el split)

Procesamiento OCR, sin nada de Normatividad:

```
OCR_TMP_DIR, OCR_OUT_DIR, OCR_LANG, OCR_MODE, OCR_DESKEW, OCR_CLEAN,
OCR_REMOVE_VECTORS, OCR_TESSERACT_PSM, OCR_JOBS, OCR_LOCAL_ROOT,
PUBLIC_BASE_URL,
OCR_MASK_STAMPS, OCR_MASK_SIGNATURES, OCR_STAMP_MIN_AREA, OCR_STAMP_MAX_AREA,
OCR_STAMP_CIRCULARITY, OCR_STAMP_RECT_ASPECT_MIN, OCR_STAMP_RECT_ASPECT_MAX,
OCR_SIGNATURE_REGION, OCR_MASK_GRAYSCALE, OCR_MASK_DILATE, STAMP_MODEL_PATH,
REVIEW_APP_VERSION, CLASSIFY_DIR, CLASSIFY_CONF_THRESHOLD, REVIEW_LOCK_TTL_MIN,
CLASSIFY_LOCK_TTL_MIN,
NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES
```

Nuevas, para el contrato `/jobs` (ver `api.md`):

```
OCR_ALLOWED_OUTPUT_ROOTS   lista separada por comas de raices de filesystem
                            montado donde /jobs puede escribir artefactos.
                            Default: OCR_OUT_DIR + OCR_SHARED_CACHE_DIR (si
                            esta definido).
OCR_JOB_STAGING_DIR        directorio de staging cuando la peticion no trae
                            artifacts.<name>.path. Default:
                            {OCR_OUT_DIR}/job_staging.
OCR_JOB_STAGING_TTL_MIN    minutos que vive un artefacto en staging antes de
                            que el sweeper lo borre. Default: 15.
OCR_JOB_WORKERS            tamano del ThreadPoolExecutor que procesa jobs.
                            Default: 2.
```

Nota GPU: `NVIDIA_*` solo aplica si `OCR_MASK_STAMPS`/`OCR_MASK_SIGNATURES`
estan activos, o se usa el detector de `text_block` — son los unicos puntos
del pipeline que tocan CUDA (`app/ocr_pipeline.py`,
`_resolve_detector_device` / `_resolve_text_block_device`).

## normatividad-ocr-adapter (repo nuevo, pendiente)

Renombradas desde `MUNIS_*`:

```
NORMATIVIDAD_BASE_URL, NORMATIVIDAD_OCR_TOKEN,
OCR_WORKER_NAME, OCR_WORKER_CONCURRENCY, OCR_POLL_INTERVAL_SECONDS,
OCR_WORKER_ENABLED, OCR_CALLBACK_TIMEOUT_SECONDS, OCR_DOWNLOAD_TIMEOUT_SECONDS,
OCR_WORKER_MAX_CONSECUTIVE_ERRORS, OCR_WORKER_LOG_LEVEL
```

Nueva:

```
OCR_CORE_URL   URL interna de ocr-core (ej. http://ocr-core:8000). Reemplaza
                el import in-process de app.ocr_pipeline.
```

Se retira: reserva de GPU (confirmado que el worker no la usa hoy — ver
`arquitectura.md`).

## ocr-search (repo nuevo, pendiente)

```
OPENSEARCH_URL, OPENSEARCH_INDEX,
OCR_SHARED_CACHE_DIR, SEARCH_INDEXER_POLL_INTERVAL_SECONDS,
SEARCH_INDEXER_BATCH_SIZE, SEARCH_INDEXER_LOG_LEVEL
```

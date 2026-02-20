# OCR Service (Pilot)

Objetivo: piloto de OCR para PDFs desde URLs, generando:
- PDF searchable
- Texto extraido
- Metricas basicas

## Enfoque
- OCR CPU: OCRmyPDF + Tesseract (estable, calidad consistente)
- Extraccion de texto: PyMuPDF
- Servicio: FastAPI

## Estructura
- app/          codigo del servicio
- data/tmp      temporales
- data/out      salidas (PDF searchable y textos)
- data/samples  muestras locales para /ocr/local
- scripts/      utilidades locales

## Variables de entorno
- OCR_TMP_DIR   (default: /data/tmp)
- OCR_OUT_DIR   (default: /data/out)
- OCR_LANG      (default: spa)
- OCR_MODE      (default: searchable_cpu)
- OCR_DESKEW (default: false)
- OCR_CLEAN (default: false)
- OCR_REMOVE_VECTORS (default: false)
- OCR_TESSERACT_PSM (opcional; ejemplo: 6)
- OCR_LOCAL_ROOT (opcional; si se define, limita /ocr/local a ese directorio)
- PUBLIC_BASE_URL (opcional; si se define, se devuelve `download_url` absoluto)
- OCR_MASK_STAMPS (default: false)
- OCR_MASK_SIGNATURES (default: false)
- OCR_STAMP_MIN_AREA (default: 20000)
- OCR_STAMP_MAX_AREA (default: 400000)
- OCR_STAMP_CIRCULARITY (default: 0.5)
- OCR_STAMP_RECT_ASPECT_MIN (default: 0.5)
- OCR_STAMP_RECT_ASPECT_MAX (default: 2.0)
- OCR_SIGNATURE_REGION (default: 0.35)

## Endpoints
- POST /ocr
  body:
  {
    "url": "https://.../archivo.pdf",
    "mode": "searchable_cpu",
    "lang": "spa"
  }

- GET /file/{filename}
  Descarga el PDF con OCR aplicado.

- POST /ocr/local
  Procesa un PDF que ya existe en el filesystem del contenedor.

Nota:
- Si se activan `mask_stamps` o `mask_signatures`, el OCR se hace sobre una version
  rasterizada con sellos/firmas enmascarados.

## Como ponerlo en funcionamiento

### Opcion 1: Docker Compose (recomendado)
Requiere una red Docker externa llamada `grh_network`.

1) Crear la red (solo una vez):
```bash
docker network create grh_network
```

2) Construir y levantar el servicio:
```bash
docker compose up -d --build
```

El servicio queda expuesto en el host en `http://localhost:18010`.

### Opcion 2: Docker directo
```bash
docker build -t ocr-service .
docker run --rm -p 18010:8000 \
  -e OCR_TMP_DIR=/data/tmp \
  -e OCR_OUT_DIR=/data/out \
  -e OCR_LANG=spa \
  -e OCR_MODE=searchable_cpu \
  -v $(pwd)/data/tmp:/data/tmp \
  -v $(pwd)/data/out:/data/out \
  --network grh_network \
  ocr-service
```

## Como usarlo

### Desde el host (puerto publicado)
```bash
curl -X POST http://localhost:18010/ocr \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://.../archivo.pdf","mode":"searchable_cpu","lang":"spa"}'
```

Luego descarga el PDF usando `download_path` o `download_url` (si `PUBLIC_BASE_URL` esta definido).

### Desde otro contenedor en la misma red Docker
Usa el nombre del servicio y el puerto interno:

```bash
curl -X POST http://ocr-service:8000/ocr \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://.../archivo.pdf","mode":"searchable_cpu","lang":"spa"}'
```

## Proximos pasos
- Agregar pipeline GPU (PaddleOCR) para texto
- Agregar creacion de PDF searchable con capa OCR GPU
- Conectar con Laravel/Django via HTTP o cola

# OCR Service

Objetivo actual:
- generar un `PDF searchable` con la imagen original intacta
- usar OCR principalmente para busqueda documental
- controlar mejor el texto falso introducido por sellos, logos, firmas y huellas

Estado del proyecto:
- servicio OCR funcional
- pipeline conservador en evolucion
- detector de sellos integrado
- linea paralela de deteccion de texto ya iniciada

## Enfoque
- OCR oficial: OCRmyPDF + Tesseract
- Salida final: PDF searchable con imagen original intacta
- Servicio: FastAPI
- Vision auxiliar:
  - detector de sellos / logos / firmas
  - preanotacion y futura deteccion de bloques de texto

## Estructura
- app/          codigo del servicio
- data/tmp      temporales
- data/out      salidas (PDF searchable y textos)
- data/samples  muestras locales para /ocr/local
- scripts/      utilidades locales

Workspaces relevantes:
- `data/out/stamp_pages` paginas usadas para sellos
- `data/out/text_pages` copia separada para trabajar cajas de texto

Split documental vigente:
- `data/train_list.txt`: 900 documentos
- `data/test_list.txt`: 100 documentos

Estado actual de listas:
- `train_list.txt` y `test_list.txt` son la definicion oficial del split
- `data/samples_train` es la descarga local del train
- `data/samples_train_list.txt` es la lista derivada de esos PDFs descargados
- `data/samples_test` es la descarga local del split de test para pruebas manuales e inferencia interactiva
- `data/sample_list.txt` queda como flujo legado / auxiliar, no como base oficial actual

## Variables de entorno
- OCR_TMP_DIR   (default: /data/tmp)
- OCR_OUT_DIR   (default: /data/out)
- OCR_LANG      (default: spa)
- OCR_MODE      (default: searchable_cpu)
- OCR_DESKEW (default: false)
- OCR_CLEAN (default: false)
- OCR_REMOVE_VECTORS (default: false)
- OCR_TESSERACT_PSM (opcional; ejemplo: 6)
- OCR_JOBS (opcional; numero de procesos para OCRmyPDF/Tesseract)
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
- OCR_MASK_GRAYSCALE (default: true; convierte a grises el PDF enmascarado)
- OCR_MASK_DILATE (default: 4; expande la mascara para cubrir mejor el sello)
- STAMP_MODEL_PATH (default efectivo: `stamp_detector_v2.pt` si existe)

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

Modo adicional:
- `searchable_conservative`
  - intenta filtrar mejor la capa OCR en regiones detectadas como sello
  - es un modo experimental de trabajo, no una solucion cerrada

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

### Interfaces de revision
- `http://localhost:18010/stamps/review`
- `http://localhost:18010/stamps/classify`
- `http://localhost:18010/text/review`
- `http://localhost:18010/text/review/skipped`
- `http://localhost:18010/text/review/compare`
- `http://localhost:18010/text/review/qc`
- `http://localhost:18010/models/test`

### Vista de prueba de modelos
- `/models/test` usa PDFs descargados en `data/samples_test`
- permite elegir PDFs de `test_list.txt`, cambiar de pagina y probar modelos en linea
- `Probar texto` corre el detector de `text_block` activo
- `Probar sellos` corre el detector de sellos activo
- las cajas se dibujan sobre la pagina renderizada sin guardar labels

### Desde otro contenedor en la misma red Docker
Usa el nombre del servicio y el puerto interno:

```bash
curl -X POST http://ocr-service:8000/ocr \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://.../archivo.pdf","mode":"searchable_cpu","lang":"spa"}'
```

## Lineas de trabajo activas
- mejorar el recorte de capa OCR en zonas de sello
- corregir manualmente cajas de `text_block`
- entrenar un primer detector de texto
- usar detector de texto + detector de sellos para decidir superposicion real

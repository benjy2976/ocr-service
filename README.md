# OCR Service

Objetivo actual:
- generar un `PDF searchable` con la imagen original intacta
- usar OCR principalmente para busqueda documental
- producir un artefacto `text` en JSONL por pagina para indexacion
- controlar mejor el texto falso introducido por sellos, logos, firmas y huellas

Estado del proyecto:
- servicio OCR funcional
- pipeline conservador en evolucion
- detector de sellos integrado
- linea paralela de deteccion de texto ya iniciada

## Enfoque
- OCR oficial: OCRmyPDF + Tesseract
- Salida final: PDF searchable con imagen original intacta
- Artefacto textual: `{source_md5}.jsonl` reportado como `artifacts.text`
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

---

## Integración Munis — ocr-worker

Este repo expone dos servicios Docker desde el mismo código base:

### ocr-api
API HTTP on-demand. Mismo comportamiento de siempre.

```bash
docker compose up ocr-api
```

### ocr-worker
Consumidor activo de la cola OCR de Normatividad en Munis.
Hace polling continuo, descarga PDFs, aplica OCR y reporta el resultado.

```bash
# Variables obligatorias antes de levantar:
export MUNIS_BASE_URL=http://munis:8000
export MUNIS_OCR_TOKEN=<token>

docker compose up ocr-worker
```

### Levantar ambos servicios

```bash
cp .env.example .env
# editar .env con los valores reales
docker compose up --build
```

### Variables de entorno del worker

| Variable                          | Default        | Descripción |
|-----------------------------------|---------------|-------------|
| MUNIS_BASE_URL                    | (obligatorio) | URL base de Munis |
| MUNIS_OCR_TOKEN                   | (obligatorio) | Bearer token de autenticación |
| OCR_WORKER_NAME                   | ocr-worker-1  | Nombre del worker en logs |
| OCR_WORKER_CONCURRENCY            | 1             | Número de consumidores paralelos dentro del mismo servicio |
| OCR_POLL_INTERVAL_SECONDS         | 10            | Segundos entre polls si la cola está vacía |
| OCR_WORKER_ENABLED                | true          | Poner "false" para deshabilitar sin borrar el contenedor |
| OCR_CALLBACK_TIMEOUT_SECONDS      | 30            | Timeout HTTP para callbacks a Munis |
| OCR_DOWNLOAD_TIMEOUT_SECONDS      | 120           | Timeout HTTP para descarga del PDF fuente |
| OCR_WORKER_MAX_CONSECUTIVE_ERRORS | 20            | Errores seguidos antes de detener el proceso |
| OCR_WORKER_LOG_LEVEL              | INFO          | Nivel de log (DEBUG/INFO/WARNING/ERROR) |

### Flujo del worker

```
pull-next (POST) → 204 → esperar → reintentar
                 → item → mark_processing (POST)
                        → download source (GET)
                        → OCR local
                        → publicar pdf/text en cache compartido
                        → artifacts (POST, JSON)       ← OK
                        → fail    (POST, JSON)         ← error
```

Si `OCR_WORKER_CONCURRENCY > 1`, un solo contenedor `ocr-worker` levanta varios
consumidores en paralelo sobre la misma cola. Cada consumidor usa un nombre
derivado de `OCR_WORKER_NAME` para que Munis y los logs distingan los leases.

### Configuración necesaria en Munis

1. Crear el token OCR y configurarlo en Munis como token válido para el servicio OCR.
2. (Opcional) Whitelist la IP del contenedor ocr-worker si Munis lo requiere.
3. Exponer los endpoints del grupo `/api/ocr/normatividad/queue/*` accesibles desde la red del worker.
4. Asegurarse de que `grh_network` es la misma red Docker que usa Munis (o ajustar la red en docker-compose.yml).

### Archivos nuevos

| Archivo                  | Descripción |
|--------------------------|-------------|
| app/munis_client.py      | Cliente HTTP para los endpoints de Munis |
| app/worker.py            | Loop de polling y lógica del worker |
| worker_entrypoint.py     | Entry point del contenedor ocr-worker |
| .env.example             | Plantilla de variables de entorno |

### Artefacto TEXT para busqueda

El artefacto textual canonico es `text`.

Ruta fisica:

```text
{artifacts_dir}/{source_md5}.jsonl
```

El worker resuelve la ruta desde `artifacts.expected.text_path`.

Formato: JSONL UTF-8, con metadatos del documento en la raiz y paginas en
`pages`.

El archivo físico se escribe como un único objeto JSON en una sola línea
terminada en `\n`. El ejemplo siguiente está expandido solo para lectura.

```json
{
  "schema": "ocr.text.document.v1",
  "page_count": 3,
  "non_empty_pages": 2,
  "text_len": 44,
  "text_source_kind": "ocr_pdf",
  "extraction_engine": "pymupdf",
  "pages": [
    {
      "page": 1,
      "text": "Texto de la pagina 1",
      "char_count": 22,
      "word_count": 5,
      "empty": false
    },
    {
      "page": 2,
      "text": "Texto de la pagina 2",
      "char_count": 22,
      "word_count": 5,
      "empty": false
    },
    {
      "page": 3,
      "text": "",
      "char_count": 0,
      "word_count": 0,
      "empty": true
    }
  ]
}
```

Reporte canonico:

```json
{"artifacts":{"text":true},"finalize_queue":true}
```

El contrato canonico del worker solo acepta `text` para el artefacto textual.

El JSONL incluye un objeto `metadata` con los IDs y campos de Munis necesarios
para busqueda:
- `regulation_file_id`
- `regulation_id`
- `source_md5`
- `pdf_path`
- `text_path`
- `reg_year`
- `reg_num`
- `reg_title`
- `reg_description`
- `regulations_tipo`
- `regulations_tipos_sigla_id`
- `regulation_type_id`
- `regulation_type_sigla_id`

Con esos datos el indexador puede responder con IDs de `files` y `regulations`
sin consultar una API adicional de Munis.

### Busqueda documental

Servicios:
- `opensearch`: motor de busqueda.
- `ocr-search-indexer`: lee `/nfs-cache/**/*.jsonl` y carga paginas a OpenSearch.
- `ocr-search-api`: expone busqueda HTTP en `http://localhost:18020`.

Levantar:

```bash
docker compose up -d --build opensearch ocr-search-api ocr-search-indexer
```

Indexar una vez manualmente:

```bash
docker compose run --rm ocr-search-indexer \
  python3 -m app.search_indexer --once
```

Buscar:

```bash
curl 'http://localhost:18020/search?q=liquidacion&year=2021'
```

La respuesta incluye `regulation_id`, archivos coincidentes, paginas, score,
rutas de artefactos y fragmentos resaltados.
Por defecto devuelve una sola vez cada norma (`group_by=regulation`), con
`matched_files` y dentro de cada archivo `matched_pages`. Para resultados por
archivo usar `group_by=file`; para resultados por página usar `group_by=page`.

Contrato para integracion con otros aplicativos:
- ver [api.md](api.md), seccion `API de busqueda`
- URL local: `http://localhost:18020`
- URL interna Docker: `http://ocr-search-api:8000`

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

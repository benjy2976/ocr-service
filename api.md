# API del microservicio OCR

Base URL (host): `http://localhost:18010`

## GET /health
Respuesta:
```json
{"status":"ok"}
```

## POST /ocr
Procesa un PDF desde una URL.

Body (JSON):
```json
{
  "url": "https://.../archivo.pdf",
  "mode": "searchable_cpu",
  "lang": "spa",
  "deskew": true,
  "clean": true,
  "remove_vectors": false,
  "psm": "6",
  "mask_stamps": true,
  "mask_signatures": true,
  "stamp_min_area": 20000,
  "stamp_max_area": 400000,
  "stamp_circularity": 0.5,
  "stamp_rect_aspect_min": 0.5,
  "stamp_rect_aspect_max": 2.0,
  "signature_region": 0.35
}
```

Respuesta (JSON):
```json
{
  "mode": "searchable_cpu",
  "lang": "spa",
  "source": "/data/tmp/xxxx.pdf",
  "output_pdf": "/data/out/xxxx_searchable.pdf",
  "content_type": "application/pdf",
  "source_bytes": 12345,
  "text_len": 6789,
  "elapsed_sec": 2.4,
  "output_filename": "xxxx_searchable.pdf",
  "download_path": "/file/xxxx_searchable.pdf",
  "download_url": "http://host:18010/file/xxxx_searchable.pdf"
}
```

Notas:
- `download_url` solo aparece si esta definido `PUBLIC_BASE_URL`.
- Si se usan `mask_stamps` o `mask_signatures`, el OCR se hace sobre una version rasterizada
  del PDF con sellos/firmas enmascarados.
- `mode=searchable_conservative` es una variante experimental para filtrar mejor
  la capa OCR en regiones detectadas como sello.

Ejemplo:
```bash
curl -X POST http://localhost:18010/ocr \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://.../archivo.pdf","mode":"searchable_cpu","lang":"spa"}'
```

## POST /ocr/local
Procesa un PDF que ya existe en el filesystem del contenedor.

Body (JSON):
```json
{
  "path": "/mnt/storage/regulations/2026/001/documento.pdf",
  "mode": "searchable_cpu",
  "lang": "spa",
  "deskew": true,
  "clean": true,
  "remove_vectors": false,
  "psm": "6"
}
```

Notas:
- Si `OCR_LOCAL_ROOT` esta definido, el path debe estar dentro de ese directorio.

## POST /ocr/file
Procesa un PDF enviado directamente.

Parametros:
- `file` (multipart/form-data): archivo PDF
- `mode` (opcional): default `searchable_cpu`
- `lang` (opcional): default `spa`
- `deskew` (opcional): true/false
- `clean` (opcional): true/false
- `remove_vectors` (opcional): true/false
- `psm` (opcional): ejemplo `6`
- `mask_stamps` (opcional): true/false
- `mask_signatures` (opcional): true/false
- `stamp_min_area` (opcional): numero
- `stamp_max_area` (opcional): numero
- `stamp_circularity` (opcional): 0..1
- `stamp_rect_aspect_min` (opcional): numero
- `stamp_rect_aspect_max` (opcional): numero
- `signature_region` (opcional): 0..1

Ejemplo:
```bash
curl -X POST http://localhost:18010/ocr/file \
  -F "file=@/ruta/archivo.pdf" \
  -F "mode=searchable_cpu" \
  -F "lang=spa"
```

Respuesta: misma estructura que `POST /ocr`.

## Interfaces web internas

### GET /stamps/review
Revision manual de cajas de sellos sobre paginas completas.

### GET /stamps/classify
Clasificacion manual de recortes de sellos / firmas / logos.

### GET /text/review
Revision manual de cajas de texto (`text_block`) tomando `labels_auto` como base
y guardando correcciones en `labels_reviewed`.

### GET /text/review/skipped
Revision final de paginas marcadas como `skipped` en la revision de texto.

### GET /text/review/compare
Comparacion interactiva entre:
- propuesta `auto`
- propuesta `modelo`
- propuesta `merge`

Notas:
- la propuesta `modelo` se genera en caliente para la imagen actual
- `merge` se calcula en backend sin una segunda corrida del modelo
- la edicion final sigue guardando en `labels_reviewed`

### GET /text/review/qc
Segundo control de paginas ya validadas.

Notas:
- usa orden estable segun `state.json`
- guarda correcciones en `labels_qc`
- no afecta la cola principal de revision

### GET /models/test
Vista web para probar modelos sobre PDFs del split de test descargados localmente.

Capacidades:
- seleccionar PDFs desde `samples_test`
- navegar por paginas
- correr inferencia en linea de:
  - modelo de texto
  - detector de sellos
- dibujar cajas sobre la pagina renderizada

Rutas asociadas:
- `GET /models/test/pdfs`
- `GET /models/test/pdf-info?pdf=...`
- `GET /models/test/page.png?pdf=...&page=...`
- `GET /models/test/infer?pdf=...&page=...&kind=text|stamps`

## Contrato de artefactos del worker

El worker consume el contrato `artifacts.requested` / `artifacts.expected` que
entrega Munis en la cola.

Ejemplo:

```json
{
  "artifacts": {
    "requested": {
      "pdf": true,
      "text": true,
      "md": false
    },
    "expected": {
      "dir": "2026/119170",
      "pdf_path": "2026/119170/hash.pdf",
      "text_path": "2026/119170/hash.json",
      "md_path": "2026/119170/hash.md"
    }
  }
}
```

`text_path` es la ruta canonica del artefacto textual.

Reporte esperado:

```json
{
  "artifacts": {
    "text": true
  },
  "finalize_queue": true
}
```

### Formato del artefacto `text`

El archivo `text` es JSON UTF-8. Los metadatos del documento van en la raiz y
el contenido de paginas va en `pages`:

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

El indexador debe consumir este archivo como fuente primaria para busqueda por
pagina o por chunks.

## GET /file/{filename}
Descarga el PDF con OCR aplicado.

Ejemplo:
```bash
curl -O http://localhost:18010/file/xxxx_searchable.pdf
```

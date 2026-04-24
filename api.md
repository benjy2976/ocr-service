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
- usa orden estable segun `state.jsonl`
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
      "text_path": "2026/119170/hash.jsonl",
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

El archivo `text` es JSONL UTF-8. Los metadatos del documento van en la raiz y
el contenido de paginas va en `pages`:

El archivo físico contiene un único objeto JSON en una sola línea terminada en
`\n`. El ejemplo está expandido para lectura.

```json
{
  "schema": "ocr.text.document.v1",
  "page_count": 3,
  "non_empty_pages": 2,
  "text_len": 44,
  "text_source_kind": "ocr_pdf",
  "extraction_engine": "pymupdf",
  "metadata": {
    "regulation_file_id": 88656,
    "regulation_id": 97539,
    "source_md5": "abcdef1234567890abcdef1234567890",
    "pdf_path": "2021/88656/abcdef1234567890abcdef1234567890.pdf",
    "text_path": "2021/88656/abcdef1234567890abcdef1234567890.jsonl",
    "reg_year": 2021,
    "reg_num": 237,
    "reg_title": "Resolucion Gerencial Regional GRI 000237-2021-GRH/GRI.",
    "reg_description": "Descripcion de la norma...",
    "regulations_tipo": 49,
    "regulations_tipos_sigla_id": 51,
    "regulation_type_id": 49,
    "regulation_type_sigla_id": 51
  },
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

## API de busqueda

Base URL local: `http://localhost:18020`

Base URL interna Docker: `http://ocr-search-api:8000`

Esta API está pensada para integrarse desde Munis u otro aplicativo. No consulta
la base de datos de Munis en tiempo real: responde desde OpenSearch usando los
metadatos guardados en el artefacto `text`.

### GET /health

Verifica que la API esté viva y que pueda conectarse a OpenSearch.

Ejemplo:

```bash
curl 'http://localhost:18020/health'
```

Respuesta:

```json
{
  "status": "ok",
  "opensearch": true,
  "index": "munis_ocr_pages"
}
```

### GET /search

Busca texto OCR indexado por página.

Método: `GET`

Content-Type de respuesta: `application/json`

Parámetros query:

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `q` | string | sí | - | Texto de búsqueda. Mínimo 1 carácter. |
| `limit` | integer | no | `10` | Cantidad de resultados. Mínimo `1`, máximo `100`. |
| `offset` | integer | no | `0` | Desplazamiento para paginación. |
| `regulation_file_id` | integer | no | - | Filtra por ID del archivo de norma. |
| `regulation_id` | integer | no | - | Filtra por ID de la norma. |
| `year` | integer | no | - | Filtra por `reg_year`. |
| `tipo` | integer | no | - | Filtra por `regulations_tipo`. Alias histórico. |
| `sigla_id` | integer | no | - | Filtra por `regulation_type_sigla_id`. Alias corto. |
| `regulation_type_id` | integer | no | - | Filtra por tipo de norma. |
| `regulation_type_sigla_id` | integer | no | - | Filtra por sigla de tipo de norma. |
| `group_by` | string | no | `regulation` | `regulation` devuelve una sola vez cada norma; `file` devuelve una vez cada archivo; `page` devuelve una fila por página encontrada. `document` se acepta como alias de `file`. |
| `matched_files_limit` | integer | no | `10` | Máximo de archivos coincidentes inspeccionados dentro de cada norma cuando `group_by=regulation`. Mínimo `1`, máximo `50`. |
| `matched_pages_limit` | integer | no | `5` | Máximo de páginas coincidentes incluidas dentro de cada archivo cuando `group_by=regulation` o `group_by=file`. Mínimo `1`, máximo `20`. |

Ejemplo:

```bash
curl 'http://localhost:18020/search?q=liquidacion&year=2021'
```

Ejemplo con filtros de tipo:

```bash
curl 'http://localhost:18020/search?q=liquidacion&regulation_type_id=49&sigla_id=51'
```

Ejemplo para devolver una fila por archivo:

```bash
curl 'http://localhost:18020/search?q=liquidacion&group_by=file'
```

Ejemplo para devolver coincidencias por página, sin agrupar:

```bash
curl 'http://localhost:18020/search?q=liquidacion&group_by=page'
```

Respuesta:

```json
{
  "query": "liquidacion",
  "group_by": "regulation",
  "total": 1,
  "total_page_matches": 7,
  "limit": 10,
  "offset": 0,
  "results": [
    {
      "regulation_file_id": 88656,
      "regulation_id": 97539,
      "source_md5": "abcdef1234567890abcdef1234567890",
      "page": 1,
      "page_count": 3,
      "score": 12.3,
      "text_path": "2021/88656/hash.jsonl",
      "pdf_path": "2021/88656/hash.pdf",
      "source_path": "2021/005/005000002372021_1619788476.pdf",
      "file_name": "Resolucion Gerencial Regional GRI 000237-2021-GRH/GRI..pdf",
      "reg_num": 237,
      "reg_year": 2021,
      "reg_date": "2021-04-29",
      "reg_title": "Resolucion Gerencial Regional GRI 000237-2021-GRH/GRI.",
      "reg_description": "Descripcion de la norma...",
      "regulations_tipo": 49,
      "regulations_tipos_sigla_id": 51,
      "regulation_type_id": 49,
      "regulation_type_sigla_id": 51,
      "text_source_kind": "ocr_pdf",
      "highlight": {
        "text": ["...<mark>Liquidación</mark> Financiera..."]
      },
      "matched_files": [
        {
          "regulation_file_id": 88656,
          "file_name": "Resolucion Gerencial Regional GRI 000237-2021-GRH/GRI..pdf",
          "pdf_path": "2021/88656/hash-a.pdf",
          "text_path": "2021/88656/hash-a.jsonl",
          "score": 12.3,
          "matched_pages": [
            {
              "page": 1,
              "char_count": 1420,
              "word_count": 210,
              "score": 12.3,
              "highlight": {
                "text": ["...<mark>Liquidación</mark> Financiera..."]
              }
            },
            {
              "page": 2,
              "char_count": 1180,
              "word_count": 180,
              "score": 8.9,
              "highlight": {
                "text": ["...gastos de <mark>Liquidación</mark>..."]
              }
            }
          ]
        },
        {
          "regulation_file_id": 88657,
          "file_name": "Anexo.pdf",
          "pdf_path": "2021/88657/hash-b.pdf",
          "text_path": "2021/88657/hash-b.jsonl",
          "score": 7.4,
          "matched_pages": [
            {
              "page": 4,
              "char_count": 980,
              "word_count": 150,
              "score": 7.4,
              "highlight": {
                "text": ["...<mark>Liquidación</mark> del anexo..."]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

Campos principales de respuesta:

- `query`: texto buscado.
- `group_by`: modo de agrupación aplicado.
- `total`: si `group_by=regulation`, total aproximado de normas encontradas;
  si `group_by=file`, total aproximado de archivos encontrados; si
  `group_by=page`, total de páginas encontradas.
- `total_page_matches`: total de páginas coincidentes antes de agrupar.
- `limit`: límite aplicado.
- `offset`: desplazamiento aplicado.
- `results`: arreglo de resultados por página indexada.
- `results[].regulation_file_id`: ID del archivo en Munis.
- `results[].regulation_id`: ID de la norma en Munis.
- `results[].page`: página principal de coincidencia. Solo aplica directamente
  cuando `group_by=page`; en modos agrupados revisar `matched_files` o
  `matched_pages`.
- `results[].score`: relevancia calculada por OpenSearch.
- `results[].highlight`: fragmentos resaltados con `<mark>...</mark>`.
- `results[].matched_files`: archivos coincidentes de la norma cuando
  `group_by=regulation`.
- `results[].matched_files[].matched_pages`: páginas coincidentes dentro de ese
  archivo.
- `results[].matched_pages`: páginas coincidentes del mismo archivo cuando
  `group_by=file`.

Notas de integración:

- Por defecto cada resultado representa una norma única (`regulation_id`).
- Si una norma tiene dos o más archivos con coincidencias, aparecerá una sola
  vez y esos archivos estarán en `matched_files`.
- Para ver cada archivo como resultado independiente, usar `group_by=file`.
- Para ver cada página como resultado independiente, usar `group_by=page`.
- `text_path` y `pdf_path` son rutas relativas al cache OCR compartido.
- La API no devuelve el texto completo de la página; devuelve metadatos y
  fragmentos resaltados.
- Si se necesita abrir el PDF desde otro aplicativo, debe construir la URL
  pública usando `pdf_path` o resolverlo desde Munis.
- El resaltado usa HTML simple con etiquetas `<mark>`.

Errores comunes:

- `422 Unprocessable Entity`: falta `q`, `limit` está fuera de rango o algún
  filtro numérico no es entero.
- `500 Internal Server Error`: OpenSearch no está disponible o el índice no se
  pudo crear/consultar.

Ejemplo de integración desde JavaScript:

```js
const params = new URLSearchParams({
  q: 'liquidacion',
  year: '2021',
  limit: '20'
});

const res = await fetch(`http://localhost:18020/search?${params}`);
const data = await res.json();

for (const item of data.results) {
  console.log(item.regulation_file_id, item.regulation_id, item.page);
}
```

## GET /file/{filename}
Descarga el PDF con OCR aplicado.

Ejemplo:
```bash
curl -O http://localhost:18010/file/xxxx_searchable.pdf
```

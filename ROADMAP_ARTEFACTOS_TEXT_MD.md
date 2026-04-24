# Hoja de Ruta: Artefactos TEXT y MD en el worker OCR

## Decisión vigente

El artefacto textual canónico es `text`.

El objetivo de `text` no es revisión humana, sino servir como fuente estructurada
para indexación documental por página y, más adelante, para derivar `md`.

Artefactos esperados:

- `pdf`: `{artifacts_dir}/{source_md5}.pdf`
- `text`: `{artifacts_dir}/{source_md5}.json`
- `md`: `{artifacts_dir}/{source_md5}.md`

La ruta principal del artefacto textual es `text_path`.

Ejemplo de payload:

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

Reporte esperado:

```json
{
  "artifacts": {
    "text": true
  },
  "finalize_queue": true
}
```

El worker acepta y reporta canónicamente `artifacts.text`.

## Formato TEXT JSON

El archivo físico de `text` es JSON UTF-8. Los metadatos del documento van en
el objeto raíz y el contenido de páginas va en el arreglo `pages`.

Ejemplo con varias páginas:

```json
{
  "schema": "ocr.text.document.v1",
  "page_count": 3,
  "non_empty_pages": 2,
  "text_len": 42,
  "text_source_kind": "ocr_pdf",
  "extraction_engine": "pymupdf",
  "pages": [
    {
      "page": 1,
      "text": "Texto de la página 1",
      "char_count": 21,
      "word_count": 5,
      "empty": false
    },
    {
      "page": 2,
      "text": "Texto de la página 2",
      "char_count": 21,
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

Campos:

- `schema`: versión del contrato del artefacto. Valor actual: `ocr.text.document.v1`.
- `page_count`: total de páginas del PDF base.
- `non_empty_pages`: total de páginas con texto no vacío.
- `text_len`: suma de caracteres extraídos en todas las páginas.
- `text_source_kind`: origen del texto.
  - `source_pdf`: PDF fuente ya tenía texto útil; no se ejecutó OCR.
  - `ocr_pdf`: texto derivado del PDF searchable generado por OCR.
  - `shared_cache_text`: artefacto `text` reutilizado desde cache.
- `extraction_engine`: motor usado para leer la capa textual. Valor actual: `pymupdf`.
- `pages`: arreglo de páginas extraídas.

Campos de cada página:

- `page`: número de página, base 1.
- `text`: texto extraído de esa página.
- `char_count`: longitud de `text`.
- `word_count`: cantidad aproximada de palabras detectadas.
- `empty`: `true` si la página no produjo texto.

Reglas:

- El archivo debe estar codificado en UTF-8.
- El worker falla si todas las páginas quedan vacías.
- Se conservan páginas vacías para mantener numeración exacta.
- El indexador debe indexar una unidad por página o dividir cada página en chunks.
- No se generan archivos por página para evitar millones de archivos pequeños en NFS.

## Flujo vigente del worker

1. Leer `artifacts.requested`.
2. Planificar solo artefactos soportados:
   - `pdf`: soportado
   - `text`: soportado
   - `md`: pendiente
3. Resolver rutas desde `artifacts.expected`.
4. Si el PDF fuente ya tiene texto útil (`signed_text` o `unsigned_text`):
   - no ejecutar OCR
   - derivar `text` desde el PDF fuente si fue solicitado
   - publicar el PDF fuente como `pdf` si también fue solicitado
5. Si el PDF no tiene texto útil:
   - ejecutar OCR
   - publicar `pdf` si fue solicitado
   - derivar `text` desde el PDF searchable si fue solicitado
6. Reportar por `/artifacts` usando nombres canónicos.

## Relación con búsqueda

El futuro `search-indexer` debe consumir `{source_md5}.json` como fuente primaria.

Flujo recomendado:

```text
ocr-worker
  -> publica pdf/text
  -> reporta artifacts.text=true
search-indexer
  -> lee text_path
  -> lee JSON
  -> recorre pages
  -> indexa por página o por chunk en OpenSearch
```

Documento sugerido para OpenSearch:

```json
{
  "document_id": 119170,
  "source_md5": "abcdef1234567890abcdef1234567890",
  "pdf_path": "2026/119170/abcdef1234567890abcdef1234567890.pdf",
  "text_path": "2026/119170/abcdef1234567890abcdef1234567890.json",
  "page": 1,
  "chunk_index": 0,
  "text": "Texto indexable...",
  "text_source_kind": "ocr_pdf"
}
```

## MD futuro

`md` debe derivarse del mismo punto de extracción que `text`.

Primera versión recomendada:

- encabezado con metadatos mínimos
- secciones por página
- párrafos simples
- sin reconstrucción compleja de tablas o columnas

`text` sigue siendo la fuente primaria para búsqueda; `md` será un artefacto para
consumo por agentes o lectura estructurada.

## Pendientes

1. Agregar pruebas automatizadas para:
   - `pdf` only
   - `text` only
   - `pdf + text`
   - cache hit de `text`
2. Implementar `md`.
3. Crear `search-indexer`.
4. Definir índice OpenSearch y estrategia de chunks.

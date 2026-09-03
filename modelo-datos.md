# Modelo de datos — artefacto TEXT

Fuente normativa unica del schema `ocr.text.document.v1`. `ROADMAP_ARTEFACTOS_TEXT_MD.md`
y `api.md` deben enlazar aqui en vez de repetir el schema completo.

## Implementacion

- `app/text_artifact.py::build_text_document()` — generico, sin conocimiento
  de Normatividad ni de ningun otro dominio.
- Usado por `app/jobs.py` (endpoint `/jobs`) y, hasta que se ejecute la Fase 2
  del split (ver `arquitectura.md`), tambien por `app/worker.py` de forma
  equivalente (logica historica, pendiente de migrar a consumir `/jobs`).

## Schema

```json
{
  "schema": "ocr.text.document.v1",
  "page_count": 3,
  "non_empty_pages": 2,
  "text_len": 42,
  "text_source_kind": "ocr_pdf",
  "extraction_engine": "pymupdf",
  "metadata": {},
  "pages": [
    { "page": 1, "text": "...", "char_count": 21, "word_count": 5, "empty": false }
  ]
}
```

Campos:

- `schema`: version del contrato. Valor actual: `ocr.text.document.v1`.
- `page_count`: total de paginas del PDF base.
- `non_empty_pages`: paginas con texto no vacio.
- `text_len`: suma de caracteres extraidos en todas las paginas.
- `text_source_kind`: `source_pdf` (PDF ya tenia texto util, no se ejecuto OCR)
  o `ocr_pdf` (texto derivado del PDF searchable generado por OCR). El worker
  historico agrega `shared_cache_text` cuando reutiliza un artefacto desde
  cache — ese valor es especifico del adaptador, no del motor.
- `extraction_engine`: motor usado para leer la capa textual. Valor actual:
  `pymupdf`.
- `metadata`: passthrough opaco. El motor OCR no le da ningun significado; lo
  llena el llamador (`options.metadata` en `POST /jobs`). Esto es lo que
  mantiene al motor generico — antes, este campo se llenaba con IDs
  especificos de Normatividad (`regulation_file_id`, `reg_year`, etc.);
  ahora esa responsabilidad es del adaptador, no del motor.
- `pages[]`: `page` (1-indexado), `text`, `char_count`, `word_count`, `empty`.

Reglas:

- UTF-8. Documento completo en una sola linea JSON terminada en `\n`.
- Se conservan paginas vacias para mantener numeracion exacta.
- Falla (`RuntimeError`) si todas las paginas quedan vacias.
- No se generan archivos por pagina (evita millones de archivos pequenos en
  storage compartido).

## Pendiente

- El uso de `metadata` con campos de Normatividad (`regulation_file_id`,
  `regulation_id`, etc.) debe documentarse en el futuro repo
  `normatividad-ocr-adapter`, no aqui — ver `arquitectura.md`.

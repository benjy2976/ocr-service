# Hoja de Ruta: Artefactos TXT y MD en el worker OCR

## Objetivo

Extender el worker para que, además de `pdf`, pueda producir `txt` primero y `md` después, siguiendo el modelo nuevo de artefactos derivados:

- `pdf`: `{artifacts_dir}/{source_md5}.pdf`
- `txt`: `{artifacts_dir}/{source_md5}.txt`
- `md`: `{artifacts_dir}/{source_md5}.md`

Siempre usando como fuente de verdad:

- `artifacts.requested`
- `artifacts.expected.*_path`

Y publicando en storage compartido (`ocr_cache` por NFS), con reporte por `/artifacts`.

## Estado actual

Hoy el worker ya quedó orientado a:

- leer `requested` y `expected`
- generar y publicar `pdf`
- reportar por `/artifacts`
- usar `complete()` solo como fallback legacy

Limitación actual:

- solo soporta `pdf`
- si no se pide `pdf`, la cola se omite
- `txt` y `md` aún no se generan

Base técnica ya disponible:

- extracción de texto desde PDF en `app/ocr_pipeline.py`, función `_extract_text(pdf_path)`

## Fase 1: Soporte inicial de TXT

### Meta

Generar `txt` a partir del PDF OCR final, sin introducir otro motor OCR ni otro pipeline.

### Decisión técnica

El `txt` inicial saldrá del `PDF searchable` final ya generado por el pipeline actual.

Ventajas:

- reutiliza el OCR actual
- reduce complejidad
- evita duplicar extracción
- mantiene consistencia entre `pdf` y `txt`

### Comportamiento esperado

1. Si la cola pide solo `pdf`:
   - comportamiento actual

2. Si la cola pide `pdf + txt`:
   - generar `pdf`
   - extraer `txt` desde ese `pdf`
   - publicar ambos
   - reportar ambos en una sola llamada a `/artifacts`

3. Si la cola pide solo `txt`:
   - generar internamente el OCR necesario
   - extraer `txt`
   - publicar solo `txt`
   - reportar solo `txt`

### Nota

Aunque la cola pida solo `txt`, al inicio conviene permitir que el worker use el pipeline PDF como paso interno técnico, sin reportar `pdf` como producido.

## Cambios previstos para TXT

### 1. Contrato interno del worker

Agregar una etapa de planificación de artefactos:

- `requested_artifacts`
- `supported_artifacts`
- `planned_artifacts`

Propuesta:

- `supported = {"pdf": true, "txt": true, "md": false}`
- `planned = requested ∩ supported`

### 2. Generación de TXT

Nueva función conceptual:

- `_generate_txt_from_pdf(pdf_path: Path) -> str`

Primera implementación:

- usar `_extract_text(pdf_path)`

### 3. Publicación de TXT

Nueva función conceptual:

- `_publish_text_to_shared_cache(content: str, target_path: Path)`

Reglas:

- escribir en `.tmp`
- renombrar con `os.replace`
- validar que el path sea relativo a `OCR_SHARED_CACHE_DIR`

### 4. Reporte por artefactos

Extender la lógica de reporte actual para soportar:

- `{"pdf": true}`
- `{"pdf": true, "txt": true}`
- `{"txt": true}`

siempre con `finalize_queue=true` solo cuando todo lo solicitado y soportado para esa cola haya sido producido.

## Fase 2: Consolidación de TXT

### Meta

Hacer el TXT más robusto e idempotente.

### Tareas

1. Validar reutilización:
   - si `expected.txt_path` ya existe y es válido, no regenerarlo

2. Evaluar reutilización de PDF ya publicado:
   - si `expected.pdf_path` ya existe, poder regenerar solo `txt`

3. Agregar validaciones mínimas:
   - archivo no vacío
   - texto con longitud razonable
   - codificación UTF-8

4. Mejorar trazabilidad en `response_payload_json`:
   - requested
   - produced
   - expected
   - paths publicados
   - métricas básicas (`text_len`, `duration_ms`)

## Fase 3: Diseño inicial de MD

### Meta

Definir `md` como artefacto pensado para consumo por agentes, no solo como `txt` renombrado.

### Principio

La primera versión de `md` debe ser simple y estable.

No intentar de inicio:

- tablas
- listas sofisticadas
- detección semántica fuerte
- reconstrucción compleja de columnas

### Formato inicial recomendado

1. Encabezado opcional con metadatos:
   - `regulation_file_id`
   - `source_md5`
   - `artifacts_dir`

2. Separación por páginas:
   - `## Página 1`
   - texto reconstruido por bloques

3. Párrafos simples y legibles

### Fuente técnica sugerida

Usar PyMuPDF sobre el PDF final:

- `page.get_text("blocks")`
- o `page.get_text("words")`

y reconstruir texto por bloques/párrafos.

## Fase 4: Implementación inicial de MD

### Cambios previstos

1. Nueva función conceptual:
   - `_generate_md_from_pdf(pdf_path: Path) -> str`

2. Publicación:
   - `_publish_text_to_shared_cache(markdown, expected.md_path)`

3. Reporte:
   - permitir `{"md": true}`
   - o combinaciones `pdf + txt + md`

### Comportamiento esperado

1. cola `pdf + md`
2. cola `txt + md`
3. cola `pdf + txt + md`

Siempre reportando solo lo realmente producido.

## Fase 5: Evolución de MD

### Meta

Mejorar la utilidad del markdown para agentes IA.

Mejoras futuras posibles:

1. detección de encabezados
2. agrupación más inteligente por bloques
3. listas simples
4. limpieza de ruido repetitivo
5. exclusión de texto claramente espurio si ya existe buena señal en OCR conservador

Esto debe ir después de tener una primera versión funcional.

## Reglas operativas a mantener

### 1. Fuente de verdad

Siempre usar:

- `artifacts.requested`
- `artifacts.expected`

No reconstruir paths por cuenta propia salvo como validación secundaria.

### 2. Publicación

Siempre publicar en storage compartido.

No depender de `ocr_path`.

### 3. Reporte

Siempre reportar por `/artifacts` cuando existan `expected.*_path`.

### 4. Idempotencia

El worker debe tolerar reproceso de un mismo `queueItem` sin romper consistencia.

### 5. Finalización

Solo cerrar la cola cuando se haya producido exactamente lo solicitado que este worker soporte y que el backend espere cerrar en esa ejecución.

## Orden recomendado de trabajo

1. implementar `txt` desde PDF final
2. publicar `txt` en NFS
3. reportar `txt` por `/artifacts`
4. soportar colas `pdf + txt`
5. soportar colas `txt` solo
6. diseñar formato mínimo de `md`
7. implementar `md` simple por bloques/páginas
8. mejorar `md` iterativamente

## Contrato interno recomendado para el worker

Funciones objetivo a futuro:

1. `_requested_artifacts(payload) -> dict`
2. `_expected_artifacts(payload) -> dict`
3. `_planned_artifacts(requested) -> dict`
4. `_generate_pdf(...) -> Path`
5. `_generate_txt_from_pdf(pdf_path) -> str`
6. `_generate_md_from_pdf(pdf_path) -> str`
7. `_publish_pdf_to_shared_cache(src, target)`
8. `_publish_text_to_shared_cache(content, target)`
9. `_report_artifacts(queue_id, produced, finalize, payloads)`

## Criterio de implementación práctica

Cuando se retome:

- primero `txt`
- luego validación e idempotencia de `txt`
- después `md`
- recién al final mejoras semánticas de `md`

Ese orden minimiza riesgo y aprovecha lo que el proyecto ya tiene hoy.

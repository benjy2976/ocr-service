# Arquitectura

## Estado actual (mono-repo)

Este repositorio hoy contiene tres dominios distintos en el mismo codebase e
imagen Docker:

| Dominio | Archivos | Servicio Docker | Generico hoy |
|---|---|---|---|
| Motor OCR + API HTTP | `app/main.py`, `app/routes_ocr.py`, `app/routes_review.py`, `app/ocr_pipeline.py`, `app/jobs.py`, `app/text_artifact.py`, `app/preflight.py` | `ocr-api` | Si |
| Adaptador Normatividad | `app/worker.py`, `app/munis_client.py`, `worker_entrypoint.py`, `worker_main.py` | `ocr-worker` | No |
| Busqueda documental | `app/search_api.py`, `app/search_indexer.py`, `app/search_common.py` | `ocr-search-api`, `ocr-search-indexer`, `opensearch` | No (esquema con campos de Normatividad: `regulation_id`, `reg_year`, etc.) |

Confirmado por inspeccion directa del codigo (no por convencion de nombres):
`app/main.py` no tiene ninguna referencia a Normatividad. El acoplamiento no
es en runtime via HTTP — es en codigo fuente: `app/worker.py` importa
`app.ocr_pipeline` **in-process** en vez de llamar al motor por HTTP.

## Plan aprobado: split en 3 repos (pendiente de ejecutar)

```
ocr-core/                    ← este repo, reducido al motor OCR generico
normatividad-ocr-adapter/    ← repo nuevo, reemplaza ocr-worker
ocr-search/                  ← repo nuevo, search_api + search_indexer + opensearch
```

Los 3 repos se crean **desde cero, sin historia heredada**. Terminologia:
"Munis" se renombra a "Normatividad" en el repo del adaptador
(`normatividad_client.py`, `NORMATIVIDAD_BASE_URL`, `NORMATIVIDAD_OCR_TOKEN`).

Decision sobre Search: se va junto con el adaptador (o repo propio pero
acoplado a su esquema) en una primera pasada. Generalizar el esquema de
busqueda (reemplazar campos fijos de Normatividad por `metadata: {}`
passthrough) queda como fase futura, no bloquea el split principal.

### Por que `normatividad-ocr-adapter` no importa el pipeline in-process

El adaptador debe llamar a `ocr-core` por HTTP (`POST /jobs`, ver `api.md`),
no importar `app.ocr_pipeline` directamente. Esto es lo que hace a `ocr-core`
reutilizable desde "muchas plataformas" (objetivo de negocio declarado), no
solo desde Normatividad.

### GPU

`ocr-worker` reservaba GPU en `docker-compose.yml` sin usarla: el worker
llama a `run_ocr_file()` sin `mask_stamps`/`mask_signatures` (default
`false`), por lo que nunca carga los detectores YOLO que son los unicos
consumidores de CUDA en el pipeline. Decision: `normatividad-ocr-adapter` no
reserva GPU. Si en el futuro necesita enmascarado de sellos, se reactiva
explicitamente ahi.

### Orden de fases

1. **Fase 0** (cerrada): contrato HTTP definido (`POST /jobs`, `GET
   /jobs/{id}`, descarga con TTL), inventario de variables de entorno por
   repo (ver `entorno.md`).
2. **Fase 1** (en curso, dentro de este repo, sin tocar infraestructura):
   - Hecho: `app/main.py` (5854 lineas) dividido en `app/routes_ocr.py` (API
     publica), `app/routes_review.py` (revision de sellos/texto/page-objects,
     ~4930 lineas) y `app/routes_classify.py` (clasificacion de sellos, ~490
     lineas). `app/classify_common.py` nuevo, unico punto compartido entre
     review y classify (`_classify_dir()`).
   - Hecho: endpoint `POST /jobs` implementado sobre `app/jobs.py` +
     `app/text_artifact.py`, reutilizando `run_ocr_file` y `preflight.py` sin
     modificarlos.
   - Verificacion aplicada al split completo: diff estructural funcion por
     funcion contra el `main.py` original en git (164 funciones/clases, todas
     presentes con cuerpo identico), conteo de decoradores de ruta (73
     originales + 3 nuevas de `/jobs` = 76, cuadra), y smoke test real en
     Docker contra los 4 modulos (`/health`, `/jobs`, `/stamps/review/total`,
     `/stamps/classify/stats`, `/page-objects/total`, todos 200). El propio
     proceso de extraccion mecanica introdujo 2 bugs reales (un decorador de
     ruta borrado de mas, otro colado en el archivo equivocado) — encontrados
     y corregidos gracias a esa verificacion, no al analisis estatico solo.
   - Pendiente: dividir `app/ocr_pipeline.py` (2549 lineas, ~90 funciones con
     dependencias cruzadas entre lo que seria "masking", "detectors" y
     "core"). Decision explicita: no se va a partir mecanicamente sin antes
     tener tests de caracterizacion — el riesgo de romper el pipeline de OCR
     en produccion es real y no hay red de seguridad automatizada hoy.
3. **Fase 2** (pendiente): extraer `normatividad-ocr-adapter`, reescribir
   `worker.py` para llamar a `ocr-core` por HTTP en vez de importar el
   pipeline.
4. **Fase 3** (pendiente): extraer `ocr-search`.
5. **Fase 4** (pendiente, este repo queda limpio): solo motor OCR + API.
6. **Fase 5** (mejoras funcionales, intercalable con lo anterior):
   - Portar deteccion de tablas de `pronatel/ocr_app.py:32-151` para
     completar el artefacto `md` (ver `ROADMAP_ARTEFACTOS_TEXT_MD.md`,
     seccion "MD futuro").
   - Resolver licencia PyMuPDF/AGPL antes de ofrecer `ocr-core` a terceros.
   - Auth ligera opt-in en `ocr-core` (API key por header) — no prioritaria
     hoy, dejar el punto de extension listo.
   - Tests de caracterizacion sobre `ocr_pipeline.py`, requisito previo a la
     Fase 1 pendiente.

## Limite conocido: job store en memoria

`app/jobs.py` guarda el estado de jobs en un diccionario en memoria del
proceso. Valido mientras `ocr-api` corra con `replicas: 1`
(`ocr-service/base/05-deployment-api.yaml`). Si se escala a mas replicas, el
estado de jobs debe moverse a un store compartido (candidato: manifiestos
JSON por `job_id` en el mismo storage compartido, siguiendo el patron ya
usado para reportes de OCR).

# Contexto de decision: OCR documental con control de sellos y texto

Fecha base: 2026-02-20  
Actualizado: 2026-03-17

## Objetivo real del producto
Aplicar OCR a un volumen grande de PDFs escaneados para busqueda documental.

El entregable principal es:
- un `PDF searchable` con la imagen original intacta
- un artefacto textual `text` en JSON por pagina para indexacion
- y, mas adelante, una salida `md` derivada para consumo estructurado

La prioridad funcional no es "limpiar visualmente" el documento sino:
- extraer texto util
- evitar inventar caracteres
- reducir al minimo el texto falso generado por sellos, logos, firmas o huellas

## Criterio de calidad
Orden de gravedad de errores:
1. peor error: agregar texto falso desde sellos / logos / firmas
2. segundo error: romper una linea de texto valida por preprocesado agresivo
3. tolerable en algunos casos: perder pocos caracteres por superposicion real del sello

## Decision tecnica base
Se mantiene `OCRmyPDF + Tesseract` como pipeline oficial de salida.

La salida final sigue siendo:
- PDF searchable con imagen original intacta
- capa OCR invisible superpuesta

La version preprocesada o enmascarada es solo un artefacto de trabajo interno.

## Por que no se cambia el pipeline oficial
- OCRmyPDF produce un PDF searchable estable y profesional
- mantiene la estructura original del PDF
- evita reconstruir manualmente el PDF final
- encaja mejor con archivo institucional y preservacion

## Cambio conceptual importante
Antes:
- la idea era "quitar sellos para que OCR lea mejor"

Ahora:
- la idea es "proteger el texto valido y bloquear texto falso"

Eso implica:
- no editar visualmente el PDF final
- usar deteccion visual como apoyo
- tomar decisiones de redaccion sobre la capa OCR, no sobre la imagen final

## Estado actual del control de sellos
### Detector principal
El detector que mejor resultado esta dando es:
- `stamp_detector_v2.pt`

### Uso del detector
Se usa para:
- detectar regiones de sello / logo / firma / huella
- generar PDFs de depuracion
- decidir que partes de la capa OCR deben recortarse

### Aprendizaje clave
La heuristica basada solo en OCR ya reconocido para inferir superposicion con texto no es suficiente.
Hace falta una mejor deteccion de texto visual real.

## Estrategia actual de OCR conservador
Existe un modo `searchable_conservative` cuya intencion es:
- partir del OCR del PDF original
- detectar cajas problemáticas
- redaccionar texto OCR dentro de esas regiones
- fusionar la capa OCR filtrada sobre el PDF original

El objetivo de este modo es reducir texto espurio sin reescribir manualmente toda la capa OCR.

## Problema abierto principal
La parte mas debil del sistema actual no es la deteccion de sellos.

La parte debil es:
- detectar con precision donde hay texto real del documento
- y, por lo tanto, decidir por que lado de una caja se puede recortar agresivamente y por cual hay que ser conservador

## Decision nueva: abrir una linea paralela de deteccion de texto
Se decide crear una linea propia para detectar texto visual en pagina.

### Motivo
Hoy estamos intentando inferir el texto visual a partir del OCR.
Eso esta al reves.

El flujo correcto debe tender a ser:
1. detectar texto visual
2. detectar sellos / logos / firmas
3. cruzar ambas capas
4. decidir redaccion OCR por interseccion real

## Dataset de texto
Se reutilizan las mismas paginas base usadas para sellos, pero en una ruta separada:
- `data/out/stamp_pages` se mantiene intacto
- `data/out/text_pages` es el nuevo workspace para texto

## Split documental oficial
El split documental vigente del proyecto es:
- `data/train_list.txt`: 900 registros
- `data/test_list.txt`: 100 registros

Estas dos listas son la definicion oficial actual de la muestra documental.

### Como se materializa ese split
- `train_list.txt` define el conjunto de PDFs usados para trabajo y entrenamiento
- esos PDFs se descargan a `data/samples_train`
- `data/samples_train_list.txt` es la lista derivada de esos PDFs ya descargados
- `test_list.txt` define el conjunto de prueba documental
- esos PDFs pueden descargarse a `data/samples_test`
- de `samples_train` salen los workspaces de paginas:
  - `data/out/stamp_pages`
  - `data/out/text_pages`

### Estado de archivos anteriores
- `data/sample_list.txt` corresponde a un flujo anterior de muestra pequena
- hoy debe considerarse un artefacto legado o auxiliar
- no es la base oficial actual del split documental

### Estructura actual
- `images`: copia de las paginas
- `labels_auto`: cajas automáticas de texto
- `labels_reviewed`: correcciones manuales
- `previews_auto`: previews de las cajas

## Preanotacion de texto
Se implemento un flujo de preanotacion con PaddleOCR sobre `text_pages/images`.

### Decision
Se empieza con una sola clase:
- `text_block`

### Motivo
- es mas rapido de anotar
- sirve para un primer modelo
- deberia bastar para tomar decisiones de cruce sello-texto

## Limitacion operativa detectada
PaddleOCR dentro del contenedor actual esta corriendo en CPU.

Confirmado:
- `compiled_with_cuda = False`
- `device = cpu`

Por eso:
- el autolabeling de texto no usa la RTX 3060
- la mejora inmediata se hizo via multiproceso CPU y modo resumible

## Decision operativa sobre autolabeling
El script de autolabel de texto debe:
- ser resumible
- no reprocesar paginas ya hechas
- reconstruir previews faltantes sin volver a correr OCR
- permitir multiproceso en CPU

## Revision humana
Ya existen dos rutas separadas:
- revision normal de texto: `/text/review`
- revision de casos saltados: `/text/review/skipped`

### Politica de trabajo
- un usuario puede `Validar`
- puede `Saltar` casos dudosos
- los `skipped` se revisan al final
- si el usuario refresca, primero debe recuperar su item `in_process`

## Vista de comparacion y control adicional
Existen vistas auxiliares separadas para no interferir con la cola principal:
- `/text/review/compare`
  - compara propuesta `auto` vs propuesta `modelo`
  - el modelo de texto se ejecuta en caliente para la imagen actual
  - `merge` se calcula en backend en el mismo request
- `/text/review/qc`
  - segundo control sobre paginas ya validadas
  - guarda correcciones en `labels_qc`
  - usa navegacion ordenada y no aleatoria

## Vista de prueba de modelos con PDFs de test
Se implemento una vista de pruebas manuales:
- `/models/test`

Objetivo:
- probar modelos sobre el split oficial de test sin tocar labels de trabajo
- navegar entre PDFs y paginas
- ejecutar inferencia en linea a voluntad

Fuente documental:
- `data/test_list.txt`
- descarga local en `data/samples_test`

Capacidades actuales:
- seleccionar PDFs desde una lista o aleatoriamente
- cambiar entre paginas del PDF
- renderizar pagina a imagen en backend
- correr el modelo activo de `text_block`
- correr el detector activo de sellos / firmas / logos
- dibujar las cajas sobre la pagina sin persistir anotaciones

Uso esperado:
- validar visualmente nuevas corridas de modelos
- comparar comportamiento sobre documentos nunca usados para entrenamiento
- inspeccionar errores antes de promover un modelo como activo

## Siguiente hito
Entrenar un primer detector de `text_block` sobre el dataset de texto.

No se recomienda entrenar aun si:
- no se reviso una muestra razonable de `labels_auto`
- no se confirmo que la calidad visual de las cajas sea aceptable

## Vision de arquitectura objetivo
Arquitectura deseada a mediano plazo:
1. detector de texto visual
2. detector de sellos / firmas / logos
3. logica de interseccion por lado / region
4. redaccion de capa OCR
5. fusion sobre PDF original

El PDF final no debe alterarse visualmente.

## Decision: separar el motor OCR del negocio (2026-09-03)

Objetivo de negocio nuevo: llevar el OCR a un servicio reutilizable desde
"muchas plataformas", no solo desde el modelo de negocio actual (cola de
Normatividad). Sin login por ahora; no es prioridad.

Diagnostico: `app/main.py` (API HTTP) ya no tenia ninguna referencia a
Normatividad — la integracion vivia aislada en `app/worker.py` +
`app/munis_client.py`, pero acoplada por **import in-process** del pipeline
(no por HTTP). `app/search_api.py` / `search_indexer.py` si tienen el esquema
acoplado a Normatividad (`regulation_id`, `reg_year`, etc.).

Decision: plan de split en 3 repos nuevos, sin historia heredada —
`ocr-core` (este repo, motor + API generica), `normatividad-ocr-adapter`
(reemplaza `ocr-worker`, llama a `ocr-core` por HTTP), `ocr-search` (se va
junto al adaptador por ahora, esquema Munis→Normatividad se generaliza
despues si hace falta). Detalle completo en `arquitectura.md`.

Terminologia: "Munis" se renombra a "Normatividad" en el repo del adaptador.

Se confirmo que `ocr-worker` reservaba GPU sin usarla (el worker no activa
`mask_stamps`/`mask_signatures`, unicos consumidores de CUDA en el
pipeline) — se retira la reserva de GPU del futuro `normatividad-ocr-adapter`.

Se diseño y aprobo el contrato `POST /jobs` (asincrono, evita timeouts en
PDFs grandes) con dos modos de entrega de artefactos: destino explicito
(`artifacts.<name>.path`, validado contra `OCR_ALLOWED_OUTPUT_ROOTS`,
escritura atomica) o staging local con TTL + descarga HTTP reintentable
(no de un solo uso, por robustez ante fallas de red). Principio aprobado: el
motor solo escribe a filesystem montado, nunca implementa clientes de
protocolo de almacenamiento (FTP, S3, etc.). Detalle en `api.md`.

Implementado en esta sesion: `app/routes_ocr.py` (API publica) y
`app/routes_review.py` (herramientas internas) separados de `app/main.py`;
`app/jobs.py` y `app/text_artifact.py` nuevos. `app/ocr_pipeline.py`
deliberadamente no se toco — sin tests de caracterizacion, partirlo ahora es
alto riesgo (ver `arquitectura.md`, seccion Fase 1).

Se evaluo adoptar la skill `gestionar-memoria-viva-proyecto` (memoria viva
con preflight/validador/markdownlint obligatorios). Decision: no adoptar la
ceremonia completa en esta etapa (no aporta valor todavia sin tests/CI
basicos), pero si mantener documentacion de contexto liviana y consultable
— este archivo, `arquitectura.md`, `entorno.md`, `modelo-datos.md`, `api.md`
— actualizada cuando cierre una decision o un incremento, sin proceso
obligatorio ni validador.

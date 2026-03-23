# Scripts

## Split documental vigente

El split oficial actual del proyecto es:
- `data/train_list.txt`: 900 documentos
- `data/test_list.txt`: 100 documentos

Convencion actual:
- `train_list.txt` define el conjunto principal de trabajo
- `download_sample_pdfs.py` descarga ese train a `data/samples_train`
- `data/samples_train_list.txt` es la lista derivada de los PDFs descargados
- `extract_stamp_pages.py` y los workspaces posteriores parten de ese train materializado
- `test_list.txt` define el conjunto oficial de prueba documental
- los PDFs de test pueden descargarse a `data/samples_test` para pruebas manuales

Estado de listas anteriores:
- `data/sample_list.txt` pertenece a una muestra previa mas pequena
- hoy debe tratarse como flujo legado o auxiliar, no como split oficial actual

## build_diverse_sample.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/build_diverse_sample.py \
  --regulations /data/samples/regulations.csv \
  --reg-files /data/samples/regulation_files.csv \
  --total 1000 \
  --test-ratio 0.1 \
  --min-per-year 5 \
  --min-per-type 5 \
  --allow-missing \
  --url-base https://proyectos.regionhuanuco.gob.pe/regulations/file
```

## build_stamp_dataset.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/build_stamp_dataset.py
```

## extract_stamp_candidates.py

Flujo legado / auxiliar.
Si se usa, normalmente toma `sample_list.txt`, no el split oficial actual.

```bash
docker compose exec -T ocr-service python3 /app/scripts/extract_stamp_candidates.py \
  --list /data/sample_list.txt \
  --out /data/out/stamp_candidates \
  --dpi 200 \
  --max-pages 1 \
  --preview
```

## extract_stamp_pages.py

Flujo oficial actual para paginas base de entrenamiento / trabajo:
- parte de `train_list.txt`
- en la practica las paginas en `stamp_pages` y `text_pages` provienen del train de 900 PDFs

```bash
docker compose exec -T ocr-service python3 /app/scripts/extract_stamp_pages.py \
  --list /data/train_list.txt \
  --out /data/out/stamp_pages \
  --dpi 200 \
  --max-pages 0 \
  --verbose \
  --log-errors
```

## download_sample_pdfs.py

Descarga local del split oficial de entrenamiento (`train_list.txt`) a `samples_train`.
El resultado esperado luego queda reflejado en `samples_train_list.txt`.

```bash
docker compose exec -T ocr-service python3 /app/scripts/download_sample_pdfs.py \
  --list /data/train_list.txt \
  --out /data/samples_train \
  --sleep 0.2 \
  --verbose
```

Descarga local del split oficial de test (`test_list.txt`) a `samples_test`.
Se usa para la vista interactiva `/models/test`.

```bash
docker compose exec -T ocr-service python3 /app/scripts/download_sample_pdfs.py \
  --list /data/test_list.txt \
  --out /data/samples_test \
  --sleep 0.2 \
  --verbose
```

## export_stamp_crops.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/export_stamp_crops.py \
  --images /data/out/stamp_pages/images \
  --labels /data/out/stamp_pages/labels \
  --out /data/classify/crops \
  --index /data/classify/index.csv
```

## build_classify_dataset.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/build_classify_dataset.py \
  --labels /data/classify/state.json \
  --crops /data/classify/crops \
  --out /data/classify/dataset \
  --val-ratio 0.1 \
  --min-per-class 50 \
  --force-include sello_completo \
  --report /data/classify/reports/classify_report.json \
  --index /data/classify/index.csv \
  --split-by source_image
```

## build_detect_dataset.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/build_detect_dataset.py \
  --pages /data/out/stamp_pages/images \
  --page-labels /data/out/stamp_pages/labels \
  --index /data/classify/index.csv \
  --classify /data/classify/state_merged.json \
  --preds /data/classify/preds.json \
  --min-conf 0.95 \
  --out /data/datasets/stamps_detect \
  --val-ratio 0.1
```

## prepare_text_pages.py

Hace una copia separada de las mismas paginas usadas para sellos, para trabajar
las cajas de texto sin tocar `stamp_pages`.

En el estado actual del proyecto, esas paginas vienen del train oficial:
- `train_list.txt` -> `samples_train` -> `stamp_pages` -> `text_pages`

```bash
docker compose exec -T ocr-service python3 /app/scripts/prepare_text_pages.py \
  --src /data/out/stamp_pages \
  --out /data/out/text_pages
```

## auto_label_text_blocks.py

Genera cajas automaticas de texto usando PaddleOCR sobre `text_pages/images`.
Puede producir una caja por linea (`--mode line`) o fusionar lineas en bloques
(`--mode block`, recomendado para empezar).

Notas operativas:
- el contenedor actual corre PaddleOCR en CPU
- el script ya es resumible: salta labels existentes
- tambien puede reconstruir previews faltantes sin volver a correr OCR
- `--workers` permite multiproceso CPU

```bash
docker compose exec -T ocr-service python3 /app/scripts/auto_label_text_blocks.py \
  --images /data/out/text_pages/images \
  --out /data/out/text_pages \
  --mode block \
  --lang es \
  --min-conf 0.4 \
  --preview \
  --workers 8
```

## build_text_detect_dataset.py

Construye el dataset YOLO para texto usando primero `labels_reviewed` y, si no
existen, cae a `labels_auto`.

```bash
docker compose exec -T ocr-service python3 /app/scripts/build_text_detect_dataset.py \
  --images /data/out/text_pages/images \
  --labels-reviewed /data/out/text_pages/labels_reviewed \
  --labels-auto /data/out/text_pages/labels_auto \
  --out /data/datasets/text_blocks_detect \
  --val-ratio 0.1
```

## Revision manual de texto

Rutas web:
- `http://localhost:18010/text/review`
- `http://localhost:18010/text/review/skipped`
- `http://localhost:18010/text/review/compare`
- `http://localhost:18010/text/review/qc`

Politica:
- `text/review` para la cola normal
- `text/review/skipped` para casos dudosos dejados al final
- las correcciones se guardan en `labels_reviewed`

## Prueba manual de modelos sobre test

Ruta web:
- `http://localhost:18010/models/test`

Requiere:
- PDFs descargados en `data/samples_test`

Permite:
- seleccionar PDFs de test
- navegar por paginas
- ejecutar en linea:
  - modelo de `text_block`
  - detector de sellos
- dibujar cajas sobre la pagina sin generar labels persistentes

## predict_classify.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/predict_classify.py \
  --model /data/classify/models/cls_v1.pt \
  --crops /data/classify/crops \
  --out /data/classify/preds.json \
  --conf 0.2
```

## Entrenamiento clasificador (recomendado)

Usa parámetros más exigentes para aprovechar GPU/VRAM cuando el dataset crezca:

- `imgsz=320` o `imgsz=384`
- `batch=64` (si la VRAM lo permite)
- `workers=4` o `8` (con `shm_size=2gb`)
- `cache=True` (si hay RAM disponible)

Ejemplo:

```bash
docker compose exec -T ocr-service yolo classify train \
  data=/data/classify/dataset \
  model=yolov8s-cls.pt \
  epochs=30 \
  imgsz=320 \
  batch=64 \
  device=0 \
  workers=4 \
  cache=True
```

## preview_pdfs.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/preview_pdfs.py \
  /data/sample_list.txt /data/out/previews 200
```

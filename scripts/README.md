# Scripts

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

```bash
docker compose exec -T ocr-service python3 /app/scripts/extract_stamp_candidates.py \
  --list /data/sample_list.txt \
  --out /data/out/stamp_candidates \
  --dpi 200 \
  --max-pages 1 \
  --preview
```

## extract_stamp_pages.py

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

```bash
docker compose exec -T ocr-service python3 /app/scripts/download_sample_pdfs.py \
  --list /data/train_list.txt \
  --out /data/samples_train \
  --sleep 0.2 \
  --verbose
```

## preview_pdfs.py

```bash
docker compose exec -T ocr-service python3 /app/scripts/preview_pdfs.py \
  /data/sample_list.txt /data/out/previews 200
```

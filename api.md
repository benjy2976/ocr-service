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

## GET /file/{filename}
Descarga el PDF con OCR aplicado.

Ejemplo:
```bash
curl -O http://localhost:18010/file/xxxx_searchable.pdf
```

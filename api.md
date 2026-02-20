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
  "lang": "spa"
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

Ejemplo:
```bash
curl -X POST http://localhost:18010/ocr \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://.../archivo.pdf","mode":"searchable_cpu","lang":"spa"}'
```

## POST /ocr/file
Procesa un PDF enviado directamente.

Parametros:
- `file` (multipart/form-data): archivo PDF
- `mode` (opcional): default `searchable_cpu`
- `lang` (opcional): default `spa`

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

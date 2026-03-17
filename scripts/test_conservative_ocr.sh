#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "Uso: $0 <url|ruta_pdf>"
  exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:18010}"

if [[ -f "$TARGET" ]]; then
  curl -s -X POST "$BASE_URL/ocr/file" \
    -F "file=@$TARGET" \
    -F "mode=searchable_conservative" \
    -F "lang=spa" \
    -F "deskew=true" \
    -F "clean=true" \
    -F "psm=6" | jq .
else
  curl -s -X POST "$BASE_URL/ocr" \
    -H 'Content-Type: application/json' \
    -d "{\"url\": \"$TARGET\", \"mode\": \"searchable_conservative\", \"lang\": \"spa\", \"deskew\": true, \"clean\": true, \"psm\": \"6\"}" | jq .
fi

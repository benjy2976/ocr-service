#!/usr/bin/env bash
set -euo pipefail

URL="$1"

curl -s -X POST http://localhost:8000/ocr \
  -H 'Content-Type: application/json' \
  -d "{\"url\": \"$URL\", \"mode\": \"searchable_cpu\", \"lang\": \"spa\"}" | jq .

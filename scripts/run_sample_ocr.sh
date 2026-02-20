#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="${1:-data/sample_list.txt}"
BASE_URL="${BASE_URL:-http://localhost:18010}"
OUT_FILE="${OUT_FILE:-data/ocr_results.jsonl}"
PARALLEL="${PARALLEL:-4}"
OCR_JOBS="${OCR_JOBS:-}"
MAX_ITEMS="${MAX_ITEMS:-}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Lista no encontrada: $LIST_FILE"
  exit 1
fi

echo "Guardando resultados en: $OUT_FILE"
: > "$OUT_FILE"

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

export BASE_URL OCR_JOBS tmp_dir

input_stream() {
  if [[ -n "$MAX_ITEMS" ]]; then
    head -n "$MAX_ITEMS" "$LIST_FILE"
  else
    cat "$LIST_FILE"
  fi
}

input_stream | while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  echo "$f"
done | xargs -I{} -P "$PARALLEL" bash -c '
  f="$1"
  path="${f/data\/samples/\/data\/samples}"
  payload="{\"path\":\"$path\",\"mode\":\"searchable_cpu\",\"lang\":\"spa\",\"deskew\":true,\"clean\":true,\"psm\":\"6\",\"mask_stamps\":true,\"mask_signatures\":true"
  if [[ -n "$OCR_JOBS" ]]; then
    payload="$payload,\"jobs\":$OCR_JOBS"
  fi
  payload="$payload}"
  out_file="$(mktemp -p "$tmp_dir" ocr_XXXXXX.json)"
  status_file="${out_file}.status"
  http_code=$(curl -s -o "$out_file" -w "%{http_code}" -X POST "$BASE_URL/ocr/local" \
    -H "Content-Type: application/json" \
    -d "$payload" \
  )
  echo "$http_code" > "$status_file"
' _ {}

cat "$tmp_dir"/*.json > "$OUT_FILE"

failed=0
for s in "$tmp_dir"/*.status; do
  [[ -f "$s" ]] || continue
  code="$(cat "$s" 2>/dev/null || true)"
  if [[ "$code" != "200" ]]; then
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "Aviso: $failed requests fallaron (HTTP != 200)."
fi

echo "Listo. Resultados en: $OUT_FILE"

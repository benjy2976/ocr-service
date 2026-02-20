#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/samples}"
COUNT="${2:-50}"
OUT_FILE="${3:-data/sample_list.txt}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Directorio no encontrado: $ROOT_DIR"
  exit 1
fi

mapfile -t files < <(find "$ROOT_DIR" -type f -iname "*.pdf")
if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No se encontraron PDFs en: $ROOT_DIR"
  exit 1
fi

printf "%s\n" "${files[@]}" | shuf -n "$COUNT" > "$OUT_FILE"
echo "Lista generada: $OUT_FILE (${COUNT} PDFs)"

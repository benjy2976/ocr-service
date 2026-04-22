"""Entry point del ocr-worker y utilidades de depuración manual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import munis_client
from app.worker import WORKER_NAME, process_single_item, run_worker, run_worker_once


def _json_dump(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Worker OCR de Normatividad y utilidades de depuración."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Inicia el worker continuo (comportamiento normal).")

    once_parser = sub.add_parser(
        "once",
        help="Hace un solo pull-next, procesa un item y sale.",
    )
    once_parser.add_argument(
        "--worker-name",
        default=WORKER_NAME,
        help="Nombre de worker a usar frente a Munis.",
    )

    pull_parser = sub.add_parser(
        "pull-next",
        help="Hace pull-next una vez y muestra el payload JSON sin procesarlo.",
    )
    pull_parser.add_argument(
        "--worker-name",
        default=WORKER_NAME,
        help="Nombre de worker a usar frente a Munis.",
    )

    processing_parser = sub.add_parser(
        "mark-processing",
        help="Marca manualmente un queue_id como processing y muestra el snapshot devuelto.",
    )
    processing_parser.add_argument("queue_id", type=int)
    processing_parser.add_argument(
        "--worker-name",
        default=WORKER_NAME,
        help="Nombre de worker a usar frente a Munis.",
    )

    download_parser = sub.add_parser(
        "download-source",
        help="Descarga manualmente el PDF fuente de un queue_id.",
    )
    download_parser.add_argument("queue_id", type=int)
    download_parser.add_argument(
        "--output",
        required=True,
        help="Ruta destino donde guardar el PDF descargado.",
    )

    process_parser = sub.add_parser(
        "process-item-json",
        help="Procesa un item JSON guardado previamente, sin hacer pull-next.",
    )
    process_parser.add_argument(
        "item_json",
        help="Archivo JSON con el item completo (idealmente el snapshot de processing).",
    )
    process_parser.add_argument(
        "--worker-name",
        default=WORKER_NAME,
        help="Nombre de worker a usar frente a Munis.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "run"):
        run_worker()
        return

    if args.command == "once":
        processed = run_worker_once(worker_name=args.worker_name)
        sys.exit(0 if processed else 2)

    if args.command == "pull-next":
        item = munis_client.pull_next(worker_name=args.worker_name)
        if item is None:
            print("null")
            sys.exit(2)
        _json_dump(item)
        return

    if args.command == "mark-processing":
        data = munis_client.mark_processing(args.queue_id, worker_name=args.worker_name)
        _json_dump(data)
        return

    if args.command == "download-source":
        output = Path(args.output)
        munis_client.download_source(args.queue_id, output)
        print(str(output))
        return

    if args.command == "process-item-json":
        path = Path(args.item_json)
        item = json.loads(path.read_text(encoding="utf-8"))
        process_single_item(item, worker_name=args.worker_name)
        return

    parser.error(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()

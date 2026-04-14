"""
Entry point para el contenedor ocr-worker.

Uso:
    python worker_entrypoint.py

El comportamiento completo está controlado por variables de entorno
(ver app/worker.py para la lista completa).
"""

from app.worker import run_worker

if __name__ == "__main__":
    run_worker()

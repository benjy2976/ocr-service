"""Ruta base del directorio de clasificacion de sellos, compartida entre
app/routes_review.py (lee el estado fusionado para el audit de revision) y
app/routes_classify.py (dueno del flujo de clasificacion en si).
"""

import os
from pathlib import Path


def _classify_dir() -> Path:
    return Path(os.getenv("CLASSIFY_DIR", "/data/classify"))

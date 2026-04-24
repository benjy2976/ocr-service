FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-spa \
    ghostscript \
    unpaper \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt \
    && python3 -m pip install --no-cache-dir paddlepaddle==2.6.2 \
    && python3 -m pip install --no-cache-dir --no-deps paddleocr==2.7.3

COPY app /app/app
COPY scripts /app/scripts
COPY worker_entrypoint.py /app/worker_entrypoint.py
COPY data/models/*.pt /app/bundled-models/stamp/
COPY data/out/text_pages/models/*.pt /app/bundled-models/text_pages/

ENV OCR_TMP_DIR=/data/tmp \
    OCR_OUT_DIR=/data/out \
    OCR_SHARED_CACHE_DIR=/nfs-cache \
    OCR_LANG=spa \
    OCR_MODE=searchable_cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

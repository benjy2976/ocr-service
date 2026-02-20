FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-spa \
    ghostscript \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV OCR_TMP_DIR=/data/tmp \
    OCR_OUT_DIR=/data/out \
    OCR_LANG=spa \
    OCR_MODE=searchable_cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

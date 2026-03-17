FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-spa \
    ghostscript \
    unpaper \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt \
    && python3 -m pip install --no-cache-dir numpy==1.26.4 opencv-python==4.10.0.84 \
    && python3 -m pip install --no-cache-dir paddlepaddle==2.6.2 \
    && python3 -m pip install --no-cache-dir --no-deps paddleocr==2.7.3

COPY app /app/app
COPY scripts /app/scripts

ENV OCR_TMP_DIR=/data/tmp \
    OCR_OUT_DIR=/data/out \
    OCR_LANG=spa \
    OCR_MODE=searchable_cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

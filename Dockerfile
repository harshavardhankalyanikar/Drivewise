# DriveWise - Metadata-Aware Automotive RAG Assistant
# Multi-purpose image: runs either the FastAPI backend or the Streamlit UI
# depending on the command passed in docker-compose.yml.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies required by PyMuPDF / pdfplumber / reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate the sample brochures so the image is demo-ready out of the box.
# (Safe to remove this line if you only want to mount your own brochures.)
RUN python scripts/generate_sample_brochures.py

RUN mkdir -p /app/data/processed /app/data/uploads /app/logs

EXPOSE 8000 8501

# Default command builds the index (if missing) then serves the API.
# Overridden by docker-compose.yml for the Streamlit service.
CMD ["sh", "-c", "python scripts/ingest.py || true; uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]

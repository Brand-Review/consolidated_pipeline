# BrandGuard consolidated_pipeline
# Used for both the FastAPI (api) and Celery worker services in docker-compose.yml

FROM python:3.11-slim

# System deps: tesseract, poppler (pdf2image), libgl (opencv), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
# requirements.txt lives inside consolidated_pipeline/ — build context is the repo root
COPY consolidated_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy consolidated_pipeline source into /app (the working dir)
COPY consolidated_pipeline/ .

# Copy sibling model modules to the absolute paths the pipeline expects
COPY ColorPaletteChecker/   /ColorPaletteChecker/
COPY FontTypographyChecker/ /FontTypographyChecker/
COPY CopywritingToneChecker/ /CopywritingToneChecker/
COPY LogoDetector/          /LogoDetector/

# Create shared directories that the API and worker both need
RUN mkdir -p /app/uploads /app/results /app/logs

# Default command (overridden per-service in docker-compose.yml)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]

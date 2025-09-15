# Multi-stage Dockerfile for BrandGuard AI Pipeline
# This Dockerfile sets up the consolidated pipeline and clones all model repositories

FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for all models
RUN pip install --no-cache-dir \
    paddlepaddle-gpu \
    paddleocr \
    vllm \
    transformers \
    torch \
    torchvision \
    ultralytics \
    opencv-python \
    scikit-learn \
    matplotlib \
    seaborn \
    pillow \
    numpy \
    pandas

# Clone all model repositories
RUN git clone https://github.com/Brand-Review/FontIdentification-PaddleOCR-gaborcselle_font_identifier.git FontTypographyChecker && \
    git clone https://github.com/Brand-Review/BrandVoiceChecker-PaddleOCR-vLLM-Qwen2.5_3B.git CopywritingToneChecker && \
    git clone https://github.com/Brand-Review/ColorExtraction-with-Kmeans-ColorSpaceProcessing-CIEDE2000DistanceColorMatching.git ColorPaletteChecker && \
    git clone https://github.com/Brand-Review/LogoDetectionModels-With-BrandPlacementRulesEngine-and-ValidationPipeline.git LogoDetector

# Copy the consolidated pipeline code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results models logs

# Set up model directories and copy models
RUN mkdir -p models/color models/typography models/copywriting models/logo

# Copy model files from cloned repositories
RUN cp -r ColorPaletteChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true && \
    cp -r FontTypographyChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true && \
    cp -r CopywritingToneChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true && \
    cp -r LogoDetector/src/* src/brandguard/ 2>/dev/null || true

# Copy configuration files
RUN cp -r ColorPaletteChecker/configs/* configs/ 2>/dev/null || true && \
    cp -r FontTypographyChecker/configs/* configs/ 2>/dev/null || true && \
    cp -r CopywritingToneChecker/configs/* configs/ 2>/dev/null || true && \
    cp -r LogoDetector/configs/* configs/ 2>/dev/null || true

# Download and setup models
RUN python -c "
import os
import sys
sys.path.append('/app')

# Download YOLO model
from ultralytics import YOLO
try:
    model = YOLO('yolov8n.pt')
    print('YOLO model downloaded successfully')
except Exception as e:
    print(f'YOLO model download failed: {e}')

# Download other models as needed
print('Model setup completed')
"

# Set permissions
RUN chmod -R 755 /app && \
    chown -R 1000:1000 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["python", "run.py"]

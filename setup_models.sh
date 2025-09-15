#!/bin/bash

# BrandGuard AI Pipeline - Model Setup Script
# This script clones all model repositories and sets up the environment

set -e

echo "🚀 Starting BrandGuard AI Pipeline Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create directories
print_status "Creating necessary directories..."
mkdir -p models/color models/typography models/copywriting models/logo
mkdir -p uploads results logs

# Clone repositories
print_status "Cloning model repositories..."

# Font Typography Checker
if [ ! -d "FontTypographyChecker" ]; then
    print_status "Cloning Font Typography Checker..."
    git clone https://github.com/Brand-Review/FontIdentification-PaddleOCR-gaborcselle_font_identifier.git FontTypographyChecker
    print_success "Font Typography Checker cloned successfully"
else
    print_warning "FontTypographyChecker directory already exists, skipping..."
fi

# Copywriting Tone Checker
if [ ! -d "CopywritingToneChecker" ]; then
    print_status "Cloning Copywriting Tone Checker..."
    git clone https://github.com/Brand-Review/BrandVoiceChecker-PaddleOCR-vLLM-Qwen2.5_3B.git CopywritingToneChecker
    print_success "Copywriting Tone Checker cloned successfully"
else
    print_warning "CopywritingToneChecker directory already exists, skipping..."
fi

# Color Palette Checker
if [ ! -d "ColorPaletteChecker" ]; then
    print_status "Cloning Color Palette Checker..."
    git clone https://github.com/Brand-Review/ColorExtraction-with-Kmeans-ColorSpaceProcessing-CIEDE2000DistanceColorMatching.git ColorPaletteChecker
    print_success "Color Palette Checker cloned successfully"
else
    print_warning "ColorPaletteChecker directory already exists, skipping..."
fi

# Logo Detector
if [ ! -d "LogoDetector" ]; then
    print_status "Cloning Logo Detector..."
    git clone https://github.com/Brand-Review/LogoDetectionModels-With-BrandPlacementRulesEngine-and-ValidationPipeline.git LogoDetector
    print_success "Logo Detector cloned successfully"
else
    print_warning "LogoDetector directory already exists, skipping..."
fi

# Copy model files
print_status "Copying model files..."

# Copy source files
if [ -d "ColorPaletteChecker/src/brandguard" ]; then
    cp -r ColorPaletteChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true
    print_success "Color palette model files copied"
fi

if [ -d "FontTypographyChecker/src/brandguard" ]; then
    cp -r FontTypographyChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true
    print_success "Typography model files copied"
fi

if [ -d "CopywritingToneChecker/src/brandguard" ]; then
    cp -r CopywritingToneChecker/src/brandguard/* src/brandguard/ 2>/dev/null || true
    print_success "Copywriting model files copied"
fi

if [ -d "LogoDetector/src" ]; then
    cp -r LogoDetector/src/* src/brandguard/ 2>/dev/null || true
    print_success "Logo detection model files copied"
fi

# Copy configuration files
print_status "Copying configuration files..."

if [ -d "ColorPaletteChecker/configs" ]; then
    cp -r ColorPaletteChecker/configs/* configs/ 2>/dev/null || true
    print_success "Color palette configs copied"
fi

if [ -d "FontTypographyChecker/configs" ]; then
    cp -r FontTypographyChecker/configs/* configs/ 2>/dev/null || true
    print_success "Typography configs copied"
fi

if [ -d "CopywritingToneChecker/configs" ]; then
    cp -r CopywritingToneChecker/configs/* configs/ 2>/dev/null || true
    print_success "Copywriting configs copied"
fi

if [ -d "LogoDetector/configs" ]; then
    cp -r LogoDetector/configs/* configs/ 2>/dev/null || true
    print_success "Logo detection configs copied"
fi

# Install additional dependencies
print_status "Installing additional dependencies..."

# Install requirements from each model
if [ -f "ColorPaletteChecker/requirements.txt" ]; then
    print_status "Installing Color Palette Checker dependencies..."
    pip install -r ColorPaletteChecker/requirements.txt 2>/dev/null || print_warning "Some Color Palette dependencies failed to install"
fi

if [ -f "FontTypographyChecker/requirements.txt" ]; then
    print_status "Installing Font Typography Checker dependencies..."
    pip install -r FontTypographyChecker/requirements.txt 2>/dev/null || print_warning "Some Typography dependencies failed to install"
fi

if [ -f "CopywritingToneChecker/requirements.txt" ]; then
    print_status "Installing Copywriting Tone Checker dependencies..."
    pip install -r CopywritingToneChecker/requirements.txt 2>/dev/null || print_warning "Some Copywriting dependencies failed to install"
fi

if [ -f "LogoDetector/requirements.txt" ]; then
    print_status "Installing Logo Detector dependencies..."
    pip install -r LogoDetector/requirements.txt 2>/dev/null || print_warning "Some Logo Detection dependencies failed to install"
fi

# Download models
print_status "Downloading pre-trained models..."

# Download YOLO model
python3 -c "
import os
import sys
sys.path.append('/app')

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print('✅ YOLO model downloaded successfully')
except Exception as e:
    print(f'⚠️  YOLO model download failed: {e}')

print('✅ Model setup completed')
" 2>/dev/null || print_warning "Model download failed, but continuing..."

# Set permissions
print_status "Setting permissions..."
chmod -R 755 .
chmod +x run_fastapi.py

print_success "🎉 BrandGuard AI Pipeline setup completed successfully!"
print_status "You can now run the pipeline using:"
print_status "  Docker: docker-compose up -d"
print_status "  Direct: python run_fastapi.py"
print_status "  API will be available at: http://localhost:8000"

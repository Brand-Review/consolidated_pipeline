#!/bin/bash

# BrandGuard AI Pipeline - Quick Start Script
# One-command deployment for immediate testing

set -e

echo "🚀 BrandGuard AI Pipeline - Quick Start"
echo "======================================"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create directories
mkdir -p uploads results models logs

# Set up environment
if [ ! -f ".env" ]; then
    cp docker.env .env
    echo "✅ Environment file created"
fi

# Build and start
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Checking service health..."
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ BrandGuard API is running!"
    echo ""
    echo "🌐 Access your API at:"
    echo "   http://localhost:8000"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "⚠️  Service may still be starting. Check logs with: docker-compose logs"
fi

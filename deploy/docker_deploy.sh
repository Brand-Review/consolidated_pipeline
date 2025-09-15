#!/bin/bash

# BrandGuard AI Pipeline - Docker Deployment Script
# This script handles the complete Docker deployment process

set -e

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Parse command line arguments
MODE=${1:-"production"}
CLEAN=${2:-"false"}

print_status "Starting BrandGuard AI Pipeline deployment in $MODE mode..."

# Clean up if requested
if [ "$CLEAN" = "true" ]; then
    print_status "Cleaning up existing containers and images..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads results models logs monitoring

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating environment file..."
    cp docker.env .env
    print_success "Environment file created"
fi

# Build and start services
print_status "Building Docker images..."
docker-compose build --no-cache

print_status "Starting services..."
if [ "$MODE" = "development" ]; then
    docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
else
    docker-compose up -d
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    print_success "BrandGuard API is healthy and running!"
else
    print_warning "BrandGuard API health check failed, but service may still be starting..."
fi

# Display service information
print_success "🎉 BrandGuard AI Pipeline deployed successfully!"
echo ""
print_status "Service URLs:"
print_status "  API: http://localhost:8000"
print_status "  Health: http://localhost:8000/api/health"
print_status "  Docs: http://localhost:8000/docs"

if [ "$MODE" = "development" ]; then
    print_status "  Grafana: http://localhost:3000 (admin/admin)"
    print_status "  Prometheus: http://localhost:9090"
fi

echo ""
print_status "Useful commands:"
print_status "  View logs: docker-compose logs -f"
print_status "  Stop services: docker-compose down"
print_status "  Restart: docker-compose restart"
print_status "  Status: docker-compose ps"

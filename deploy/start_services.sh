#!/bin/bash

# BrandGuard Services Startup Script
# Starts all required services for BrandGuard API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if services are already running
check_running_services() {
    print_status "Checking for running services..."
    
    # Check API server
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        print_warning "API server is already running on port 5001"
        return 1
    fi
    
    # Check vLLM server
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_warning "vLLM server is already running on port 8000"
        return 1
    fi
    
    return 0
}

# Start vLLM server
start_vllm() {
    print_status "Starting vLLM server..."
    
    # Check if we're in the right directory
    if [ ! -f "../LogoDetector/setup_vllm.py" ]; then
        print_error "LogoDetector directory not found. Please run from consolidated_pipeline directory."
        exit 1
    fi
    
    # Start vLLM in background
    cd ../LogoDetector
    nohup python setup_vllm.py > ../consolidated_pipeline/logs/vllm.log 2>&1 &
    VLLM_PID=$!
    echo $VLLM_PID > ../consolidated_pipeline/logs/vllm.pid
    
    # Wait for vLLM to start
    print_status "Waiting for vLLM server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            print_success "vLLM server started successfully (PID: $VLLM_PID)"
            break
        fi
        echo -n "."
        sleep 10
    done
    
    if [ $i -eq 30 ]; then
        print_error "vLLM server failed to start within 5 minutes"
        exit 1
    fi
    
    cd ../consolidated_pipeline
}

# Start BrandGuard API
start_api() {
    print_status "Starting BrandGuard API server..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start API server in background
    nohup python run.py --host 0.0.0.0 --port 5001 > logs/api.log 2>&1 &
    API_PID=$!
    echo $API_PID > logs/api.pid
    
    # Wait for API to start
    print_status "Waiting for API server to start..."
    for i in {1..10}; do
        if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
            print_success "BrandGuard API started successfully (PID: $API_PID)"
            break
        fi
        echo -n "."
        sleep 5
    done
    
    if [ $i -eq 10 ]; then
        print_error "API server failed to start within 1 minute"
        exit 1
    fi
}

# Display service information
show_service_info() {
    echo
    print_success "🎉 All services started successfully!"
    echo
    echo "📊 Service Information:"
    echo "======================"
    echo "🌐 BrandGuard API: http://localhost:5001"
    echo "🤖 vLLM Server: http://localhost:8000"
    echo "📝 API Logs: logs/api.log"
    echo "📝 vLLM Logs: logs/vllm.log"
    echo
    echo "🔍 Health Checks:"
    echo "curl http://localhost:5001/api/health"
    echo "curl http://localhost:8000/v1/models"
    echo
    echo "🛑 To stop services: ./deploy/stop_services.sh"
    echo "📊 To check status: ./deploy/status.sh"
    echo
}

# Main function
main() {
    echo "🚀 Starting BrandGuard Services"
    echo "==============================="
    echo
    
    # Check if services are already running
    if ! check_running_services; then
        print_warning "Some services are already running. Use ./deploy/stop_services.sh to stop them first."
        exit 1
    fi
    
    # Start services
    start_vllm
    start_api
    
    # Show service information
    show_service_info
}

# Run main function
main "$@"

#!/bin/bash

# BrandGuard Services Stop Script
# Stops all BrandGuard services

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

# Stop API server
stop_api() {
    print_status "Stopping BrandGuard API server..."
    
    if [ -f "logs/api.pid" ]; then
        API_PID=$(cat logs/api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            print_success "API server stopped (PID: $API_PID)"
        else
            print_warning "API server was not running"
        fi
        rm -f logs/api.pid
    else
        print_warning "No API PID file found"
    fi
    
    # Kill any remaining processes on port 5001
    if lsof -ti:5001 > /dev/null 2>&1; then
        print_status "Killing remaining processes on port 5001..."
        lsof -ti:5001 | xargs kill -9 2>/dev/null || true
    fi
}

# Stop vLLM server
stop_vllm() {
    print_status "Stopping vLLM server..."
    
    if [ -f "logs/vllm.pid" ]; then
        VLLM_PID=$(cat logs/vllm.pid)
        if kill -0 $VLLM_PID 2>/dev/null; then
            kill $VLLM_PID
            print_success "vLLM server stopped (PID: $VLLM_PID)"
        else
            print_warning "vLLM server was not running"
        fi
        rm -f logs/vllm.pid
    else
        print_warning "No vLLM PID file found"
    fi
    
    # Kill any remaining processes on port 8000
    if lsof -ti:8000 > /dev/null 2>&1; then
        print_status "Killing remaining processes on port 8000..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    fi
}

# Check if services are stopped
check_services_stopped() {
    print_status "Verifying services are stopped..."
    
    # Check API server
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        print_error "API server is still running"
        return 1
    else
        print_success "API server is stopped"
    fi
    
    # Check vLLM server
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_error "vLLM server is still running"
        return 1
    else
        print_success "vLLM server is stopped"
    fi
    
    return 0
}

# Main function
main() {
    echo "🛑 Stopping BrandGuard Services"
    echo "==============================="
    echo
    
    # Stop services
    stop_api
    stop_vllm
    
    # Verify services are stopped
    if check_services_stopped; then
        print_success "🎉 All services stopped successfully!"
    else
        print_error "Some services may still be running. Check manually."
        exit 1
    fi
    
    echo
    echo "📊 Service Status:"
    echo "=================="
    echo "🌐 BrandGuard API: Stopped"
    echo "🤖 vLLM Server: Stopped"
    echo
    echo "🚀 To start services again: ./deploy/start_services.sh"
    echo
}

# Run main function
main "$@"

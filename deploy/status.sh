#!/bin/bash

# BrandGuard Services Status Script
# Shows the status of all BrandGuard services

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

# Check API server status
check_api_status() {
    print_status "Checking BrandGuard API server..."
    
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        # Get API response
        API_RESPONSE=$(curl -s http://localhost:5001/api/health)
        print_success "API server is running on port 5001"
        
        # Parse and display API status
        if echo "$API_RESPONSE" | grep -q "healthy"; then
            print_success "API health check: PASSED"
        else
            print_warning "API health check: UNKNOWN"
        fi
        
        # Check PID file
        if [ -f "logs/api.pid" ]; then
            API_PID=$(cat logs/api.pid)
            if kill -0 $API_PID 2>/dev/null; then
                print_success "API PID: $API_PID"
            else
                print_warning "PID file exists but process not found"
            fi
        fi
        
        return 0
    else
        print_error "API server is not responding on port 5001"
        return 1
    fi
}

# Check vLLM server status
check_vllm_status() {
    print_status "Checking vLLM server..."
    
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_success "vLLM server is running on port 8000"
        
        # Get model list
        MODELS_RESPONSE=$(curl -s http://localhost:8000/v1/models)
        if echo "$MODELS_RESPONSE" | grep -q "Qwen"; then
            print_success "Qwen2.5-VL-3B-Instruct model is loaded"
        else
            print_warning "Model status unknown"
        fi
        
        # Check PID file
        if [ -f "logs/vllm.pid" ]; then
            VLLM_PID=$(cat logs/vllm.pid)
            if kill -0 $VLLM_PID 2>/dev/null; then
                print_success "vLLM PID: $VLLM_PID"
            else
                print_warning "PID file exists but process not found"
            fi
        fi
        
        return 0
    else
        print_error "vLLM server is not responding on port 8000"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Memory usage
    if command -v free &> /dev/null; then
        MEMORY_USAGE=$(free -h | grep "Mem:" | awk '{print $3 "/" $2}')
        print_status "Memory usage: $MEMORY_USAGE"
    fi
    
    # Disk usage
    if command -v df &> /dev/null; then
        DISK_USAGE=$(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
        print_status "Disk usage: $DISK_USAGE"
    fi
    
    # GPU usage (if available)
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            print_status "  $line"
        done
    fi
}

# Check log files
check_logs() {
    print_status "Checking log files..."
    
    # API logs
    if [ -f "logs/api.log" ]; then
        LOG_SIZE=$(du -h logs/api.log | cut -f1)
        print_status "API log size: $LOG_SIZE"
        
        # Show last few lines
        print_status "Recent API log entries:"
        tail -3 logs/api.log | sed 's/^/  /'
    else
        print_warning "API log file not found"
    fi
    
    # vLLM logs
    if [ -f "logs/vllm.log" ]; then
        LOG_SIZE=$(du -h logs/vllm.log | cut -f1)
        print_status "vLLM log size: $LOG_SIZE"
        
        # Show last few lines
        print_status "Recent vLLM log entries:"
        tail -3 logs/vllm.log | sed 's/^/  /'
    else
        print_warning "vLLM log file not found"
    fi
}

# Check port usage
check_ports() {
    print_status "Checking port usage..."
    
    # Check port 5001
    if lsof -ti:5001 > /dev/null 2>&1; then
        PORT_5001_PID=$(lsof -ti:5001)
        print_status "Port 5001: Used by PID $PORT_5001_PID"
    else
        print_status "Port 5001: Available"
    fi
    
    # Check port 8000
    if lsof -ti:8000 > /dev/null 2>&1; then
        PORT_8000_PID=$(lsof -ti:8000)
        print_status "Port 8000: Used by PID $PORT_8000_PID"
    else
        print_status "Port 8000: Available"
    fi
}

# Main function
main() {
    echo "📊 BrandGuard Services Status"
    echo "============================="
    echo
    
    # Check services
    API_STATUS=0
    VLLM_STATUS=0
    
    check_api_status || API_STATUS=1
    echo
    check_vllm_status || VLLM_STATUS=1
    echo
    
    # Check system resources
    check_system_resources
    echo
    
    # Check logs
    check_logs
    echo
    
    # Check ports
    check_ports
    echo
    
    # Summary
    echo "📋 Summary:"
    echo "==========="
    if [ $API_STATUS -eq 0 ]; then
        print_success "BrandGuard API: RUNNING"
    else
        print_error "BrandGuard API: STOPPED"
    fi
    
    if [ $VLLM_STATUS -eq 0 ]; then
        print_success "vLLM Server: RUNNING"
    else
        print_error "vLLM Server: STOPPED"
    fi
    
    echo
    if [ $API_STATUS -eq 0 ] && [ $VLLM_STATUS -eq 0 ]; then
        print_success "🎉 All services are running properly!"
        echo
        echo "🌐 Web Interface: http://localhost:5001"
        echo "🔍 Health Check: curl http://localhost:5001/api/health"
    else
        print_error "❌ Some services are not running"
        echo
        echo "🚀 To start services: ./deploy/start_services.sh"
        echo "🛑 To stop services: ./deploy/stop_services.sh"
    fi
    echo
}

# Run main function
main "$@"

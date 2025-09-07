#!/bin/bash

# BrandGuard Quick Deploy Script
# One-click deployment for BrandGuard API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    BrandGuard Quick Deploy                  ║"
    echo "║              Easy API Deployment Without Docker             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

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

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | grep "Mem:" | awk '{print $2}')
        if [ $MEMORY_GB -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        fi
    fi
    
    print_success "System requirements check passed"
}

# Setup deployment
setup_deployment() {
    print_status "Setting up deployment..."
    
    # Run setup script
    if [ -f "deploy/setup.sh" ]; then
        chmod +x deploy/setup.sh
        ./deploy/setup.sh
    else
        print_error "Setup script not found"
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting BrandGuard services..."
    
    # Start vLLM server in background
    print_status "Starting vLLM server (this may take a few minutes)..."
    nohup ./deploy/start_services.sh > logs/deploy.log 2>&1 &
    DEPLOY_PID=$!
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    for i in {1..60}; do
        if curl -s http://localhost:5001/api/health > /dev/null 2>&1 && \
           curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            print_success "All services started successfully!"
            break
        fi
        echo -n "."
        sleep 5
    done
    
    if [ $i -eq 60 ]; then
        print_error "Services failed to start within 5 minutes"
        print_status "Check logs/deploy.log for details"
        exit 1
    fi
}

# Test deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Test API health
    if curl -s http://localhost:5001/api/health | grep -q "healthy"; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        return 1
    fi
    
    # Test vLLM
    if curl -s http://localhost:8000/v1/models | grep -q "Qwen"; then
        print_success "vLLM model check passed"
    else
        print_error "vLLM model check failed"
        return 1
    fi
    
    return 0
}

# Show deployment info
show_deployment_info() {
    echo
    print_success "🎉 BrandGuard deployment completed successfully!"
    echo
    echo "📊 Deployment Information:"
    echo "========================="
    echo "🌐 Web Interface: http://localhost:5001"
    echo "🔗 API Endpoint: http://localhost:5001/api/analyze"
    echo "🤖 vLLM Server: http://localhost:8000"
    echo "📝 Logs Directory: logs/"
    echo
    echo "🔍 Quick Tests:"
    echo "==============="
    echo "curl http://localhost:5001/api/health"
    echo "curl -X POST http://localhost:5001/api/analyze -F 'file=@test_image.jpg'"
    echo
    echo "📚 Management Commands:"
    echo "======================="
    echo "Check Status: ./deploy/status.sh"
    echo "Stop Services: ./deploy/stop_services.sh"
    echo "Start Services: ./deploy/start_services.sh"
    echo "View Logs: tail -f logs/api.log"
    echo
    echo "🚀 Your BrandGuard API is ready to use!"
    echo
}

# Main deployment function
main() {
    print_banner
    echo
    
    # Check if already running
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        print_warning "BrandGuard API is already running"
        echo "Stop it first with: ./deploy/stop_services.sh"
        exit 1
    fi
    
    # Run deployment steps
    check_requirements
    setup_deployment
    start_services
    
    # Test deployment
    if test_deployment; then
        show_deployment_info
    else
        print_error "Deployment test failed"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "BrandGuard Quick Deploy Script"
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --setup-only   Only run setup, don't start services"
        echo "  --test-only    Only test existing deployment"
        echo
        exit 0
        ;;
    --setup-only)
        print_banner
        check_requirements
        setup_deployment
        print_success "Setup completed. Run without --setup-only to start services."
        exit 0
        ;;
    --test-only)
        if test_deployment; then
            print_success "Deployment test passed"
        else
            print_error "Deployment test failed"
            exit 1
        fi
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

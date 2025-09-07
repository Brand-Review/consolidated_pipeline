#!/bin/bash

# BrandGuard Easy Deployment Setup Script
# This script sets up everything needed to run BrandGuard API

set -e  # Exit on any error

echo "🚀 BrandGuard Easy Deployment Setup"
echo "=================================="

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip first."
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs
    mkdir -p uploads
    mkdir -p results
    mkdir -p configs
    print_success "Directories created"
}

# Create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Create production config if it doesn't exist
    if [ ! -f "configs/production.yaml" ]; then
        cat > configs/production.yaml << EOF
server:
  host: "0.0.0.0"
  port: 5001
  debug: false
  workers: 4

models:
  vllm:
    enabled: true
    host: "localhost"
    port: 8000
    timeout: 120
  
  color_analysis:
    enabled: true
    n_colors: 8
    n_clusters: 8
  
  typography:
    enabled: true
    confidence_threshold: 0.7
  
  copywriting:
    enabled: true
    formality_score: 60
  
  logo_detection:
    enabled: true
    confidence_threshold: 0.5

logging:
  level: "INFO"
  file: "logs/api.log"
EOF
        print_success "Production configuration created"
    fi
    
    # Create environment file
    if [ ! -f ".env" ]; then
        cat > .env << EOF
BRANDGUARD_HOST=0.0.0.0
BRANDGUARD_PORT=5001
BRANDGUARD_DEBUG=false
VLLM_HOST=localhost
VLLM_PORT=8000
LOG_LEVEL=INFO
EOF
        print_success "Environment file created"
    fi
}

# Create startup scripts
create_scripts() {
    print_status "Creating startup scripts..."
    
    # Create start script
    cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting BrandGuard API..."

# Activate virtual environment
source venv/bin/activate

# Start the API server
python run.py --host 0.0.0.0 --port 5001

echo "✅ BrandGuard API started on http://localhost:5001"
EOF
    chmod +x start.sh
    
    # Create vLLM start script
    cat > start_vllm.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting vLLM server..."

# Activate virtual environment
source venv/bin/activate

# Start vLLM server
cd ../LogoDetector
python setup_vllm.py

echo "✅ vLLM server started on http://localhost:8000"
EOF
    chmod +x start_vllm.sh
    
    # Create health check script
    cat > health_check.sh << 'EOF'
#!/bin/bash
echo "🔍 Checking BrandGuard API health..."

# Check API health
if curl -s http://localhost:5001/api/health > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
fi

# Check vLLM health
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "✅ vLLM server is healthy"
else
    echo "❌ vLLM server is not responding"
fi
EOF
    chmod +x health_check.sh
    
    print_success "Startup scripts created"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from brandguard.config.settings import Settings
    from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
    print('✅ Core modules imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    print_success "Installation test passed"
}

# Main setup function
main() {
    echo
    print_status "Starting BrandGuard setup process..."
    echo
    
    # Run setup steps
    check_python
    check_pip
    setup_venv
    install_dependencies
    create_directories
    create_configs
    create_scripts
    test_installation
    
    echo
    print_success "🎉 BrandGuard setup completed successfully!"
    echo
    echo "📋 Next steps:"
    echo "1. Start vLLM server: ./start_vllm.sh (in a separate terminal)"
    echo "2. Start BrandGuard API: ./start.sh"
    echo "3. Test the API: ./health_check.sh"
    echo "4. Open web interface: http://localhost:5001"
    echo
    echo "📚 For more information, see EASY_DEPLOYMENT_GUIDE.md"
    echo
}

# Run main function
main "$@"

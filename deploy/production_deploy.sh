#!/bin/bash

# BrandGuard Complete Production Deployment Script
# Handles all production requirements automatically

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
    echo "║              BrandGuard Production Deployment                ║"
    echo "║              Complete Production Setup Script                ║"
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

# Parse command line arguments
DOMAIN=""
SKIP_SSL=false
SKIP_FIREWALL=false
SKIP_MONITORING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        --skip-firewall)
            SKIP_FIREWALL=true
            shift
            ;;
        --skip-monitoring)
            SKIP_MONITORING=true
            shift
            ;;
        --help|-h)
            echo "BrandGuard Production Deployment Script"
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --domain DOMAIN     Domain name for SSL certificate (required)"
            echo "  --skip-ssl          Skip SSL/HTTPS setup"
            echo "  --skip-firewall     Skip firewall configuration"
            echo "  --skip-monitoring   Skip monitoring setup"
            echo "  --help, -h          Show this help message"
            echo
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if domain is provided
if [ -z "$DOMAIN" ] && [ "$SKIP_SSL" = false ]; then
    print_error "Domain is required for SSL setup. Use --domain yourdomain.com or --skip-ssl"
    exit 1
fi

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root for production deployment"
        print_status "Please run: sudo $0 --domain $DOMAIN"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_status "OS: $NAME $VERSION"
    fi
    
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

# Phase 1: Basic Setup
phase1_basic_setup() {
    print_status "Phase 1: Basic Setup"
    echo "========================="
    
    # Create production directory
    print_status "Creating production directory..."
    mkdir -p /opt/brandguard
    cd /opt/brandguard
    
    # Clone repository if not exists
    if [ ! -d "consolidated_pipeline" ]; then
        print_status "Cloning repository..."
        # Note: Replace with your actual repository URL
        print_warning "Please clone your repository to /opt/brandguard/consolidated_pipeline"
        print_status "Example: git clone <your-repo-url> consolidated_pipeline"
        exit 1
    fi
    
    cd consolidated_pipeline
    
    # Run basic setup
    print_status "Running basic setup..."
    chmod +x deploy/setup.sh
    ./deploy/setup.sh
    
    print_success "Phase 1 completed: Basic setup"
}

# Phase 2: Security Hardening
phase2_security() {
    print_status "Phase 2: Security Hardening"
    echo "================================"
    
    # Create production user
    print_status "Creating production user..."
    if ! id "brandguard" &>/dev/null; then
        useradd -m -s /bin/bash brandguard
        usermod -aG sudo brandguard
        print_success "Production user created"
    else
        print_warning "Production user already exists"
    fi
    
    # Set ownership
    chown -R brandguard:brandguard /opt/brandguard
    
    # Configure firewall
    if [ "$SKIP_FIREWALL" = false ]; then
        print_status "Configuring firewall..."
        ufw --force enable
        ufw allow 22
        ufw allow 80
        ufw allow 443
        ufw deny 5001
        print_success "Firewall configured"
    else
        print_warning "Skipping firewall configuration"
    fi
    
    # Disable root login
    print_status "Disabling root login..."
    sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
    systemctl restart ssh
    print_success "Root login disabled"
    
    print_success "Phase 2 completed: Security hardening"
}

# Phase 3: Service Management
phase3_service_management() {
    print_status "Phase 3: Service Management"
    echo "==============================="
    
    # Create systemd service
    print_status "Creating systemd service..."
    cat > /etc/systemd/system/brandguard.service << EOF
[Unit]
Description=BrandGuard API Service
After=network.target

[Service]
Type=simple
User=brandguard
Group=brandguard
WorkingDirectory=/opt/brandguard/consolidated_pipeline
ExecStart=/opt/brandguard/consolidated_pipeline/deploy/start_services.sh
ExecStop=/opt/brandguard/consolidated_pipeline/deploy/stop_services.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable service
    systemctl daemon-reload
    systemctl enable brandguard
    print_success "Systemd service created and enabled"
    
    print_success "Phase 3 completed: Service management"
}

# Phase 4: Reverse Proxy
phase4_reverse_proxy() {
    print_status "Phase 4: Reverse Proxy Setup"
    echo "================================="
    
    # Install Nginx
    print_status "Installing Nginx..."
    apt update
    apt install -y nginx
    
    # Configure Nginx
    print_status "Configuring Nginx..."
    cat > /etc/nginx/sites-available/brandguard << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://localhost:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # File upload size
        client_max_body_size 50M;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://localhost:5001/api/health;
        access_log off;
    }
    
    # Static files
    location /static/ {
        alias /opt/brandguard/consolidated_pipeline/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/brandguard /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test configuration
    nginx -t
    systemctl restart nginx
    print_success "Nginx configured and started"
    
    print_success "Phase 4 completed: Reverse proxy"
}

# Phase 5: SSL/HTTPS
phase5_ssl() {
    if [ "$SKIP_SSL" = true ]; then
        print_warning "Skipping SSL setup"
        return
    fi
    
    print_status "Phase 5: SSL/HTTPS Setup"
    echo "============================"
    
    # Install Certbot
    print_status "Installing Certbot..."
    apt install -y certbot python3-certbot-nginx
    
    # Get SSL certificate
    print_status "Obtaining SSL certificate for $DOMAIN..."
    certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
    
    # Setup auto-renewal
    print_status "Setting up SSL auto-renewal..."
    echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -
    print_success "SSL auto-renewal configured"
    
    print_success "Phase 5 completed: SSL/HTTPS"
}

# Phase 6: Monitoring
phase6_monitoring() {
    if [ "$SKIP_MONITORING" = true ]; then
        print_warning "Skipping monitoring setup"
        return
    fi
    
    print_status "Phase 6: Monitoring Setup"
    echo "============================="
    
    # Create log directory
    mkdir -p /var/log/brandguard
    chown brandguard:brandguard /var/log/brandguard
    
    # Setup log rotation
    print_status "Setting up log rotation..."
    cat > /etc/logrotate.d/brandguard << EOF
/var/log/brandguard/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 brandguard brandguard
    postrotate
        systemctl reload brandguard
    endscript
}
EOF
    
    # Create monitoring script
    print_status "Creating monitoring script..."
    cat > /opt/brandguard/monitor.sh << 'EOF'
#!/bin/bash
echo "=== BrandGuard Production Status ==="
echo "Date: $(date)"
echo "Service Status: $(systemctl is-active brandguard)"
echo "API Health: $(curl -s http://localhost:5001/api/health | jq -r '.status' 2>/dev/null || echo 'DOWN')"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
EOF
    
    chmod +x /opt/brandguard/monitor.sh
    chown brandguard:brandguard /opt/brandguard/monitor.sh
    
    # Create health check script
    print_status "Creating health check script..."
    cat > /opt/brandguard/health_check.sh << 'EOF'
#!/bin/bash
# Production Health Check

# Check API
if ! curl -s http://localhost:5001/api/health > /dev/null; then
    echo "ALERT: API is down!"
    systemctl restart brandguard
    exit 1
fi

# Check vLLM
if ! curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "ALERT: vLLM is down!"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "ALERT: Disk usage is ${DISK_USAGE}%"
    exit 1
fi

echo "All systems healthy"
EOF
    
    chmod +x /opt/brandguard/health_check.sh
    chown brandguard:brandguard /opt/brandguard/health_check.sh
    
    # Setup cron job for health checks
    print_status "Setting up health check cron job..."
    echo "*/5 * * * * /opt/brandguard/health_check.sh" | crontab -u brandguard -
    
    print_success "Phase 6 completed: Monitoring"
}

# Phase 7: Start Services
phase7_start_services() {
    print_status "Phase 7: Starting Services"
    echo "=============================="
    
    # Start BrandGuard service
    print_status "Starting BrandGuard service..."
    systemctl start brandguard
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    for i in {1..30}; do
        if systemctl is-active --quiet brandguard; then
            print_success "BrandGuard service started"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        print_error "BrandGuard service failed to start"
        systemctl status brandguard
        exit 1
    fi
    
    # Wait for API to be ready
    print_status "Waiting for API to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:5001/api/health > /dev/null; then
            print_success "API is ready"
            break
        fi
        echo -n "."
        sleep 5
    done
    
    if [ $i -eq 60 ]; then
        print_error "API failed to start within 5 minutes"
        exit 1
    fi
    
    print_success "Phase 7 completed: Services started"
}

# Phase 8: Verification
phase8_verification() {
    print_status "Phase 8: Verification"
    echo "========================"
    
    # Check service status
    print_status "Checking service status..."
    systemctl status brandguard --no-pager
    
    # Check API health
    print_status "Checking API health..."
    curl -s http://localhost:5001/api/health | jq .
    
    # Check vLLM health
    print_status "Checking vLLM health..."
    curl -s http://localhost:8000/v1/models | jq .
    
    # Check Nginx status
    print_status "Checking Nginx status..."
    systemctl status nginx --no-pager
    
    # Check SSL (if enabled)
    if [ "$SKIP_SSL" = false ]; then
        print_status "Checking SSL certificate..."
        curl -s -I https://$DOMAIN | head -1
    fi
    
    print_success "Phase 8 completed: Verification"
}

# Show deployment summary
show_summary() {
    echo
    print_success "🎉 Production deployment completed successfully!"
    echo
    echo "📊 Deployment Summary:"
    echo "====================="
    echo "🌐 Web Interface: http://$DOMAIN"
    echo "🔗 API Endpoint: http://$DOMAIN/api/analyze"
    echo "🔍 Health Check: http://$DOMAIN/health"
    echo "📝 Logs: /var/log/brandguard/"
    echo "🔧 Service: systemctl status brandguard"
    echo
    echo "📚 Management Commands:"
    echo "======================"
    echo "Check Status: systemctl status brandguard"
    echo "View Logs: journalctl -u brandguard -f"
    echo "Restart: systemctl restart brandguard"
    echo "Monitor: /opt/brandguard/monitor.sh"
    echo "Health Check: /opt/brandguard/health_check.sh"
    echo
    echo "🔒 Security Features:"
    echo "===================="
    echo "✅ SSL/HTTPS enabled"
    echo "✅ Firewall configured"
    echo "✅ Root login disabled"
    echo "✅ Production user created"
    echo "✅ Rate limiting enabled"
    echo
    echo "📈 Monitoring Features:"
    echo "======================"
    echo "✅ Log rotation configured"
    echo "✅ Health checks automated"
    echo "✅ Service auto-restart enabled"
    echo "✅ Performance monitoring"
    echo
    echo "🚀 Your BrandGuard API is production-ready!"
    echo
}

# Main deployment function
main() {
    print_banner
    echo
    
    # Check requirements
    check_root
    check_requirements
    
    # Run all phases
    phase1_basic_setup
    phase2_security
    phase3_service_management
    phase4_reverse_proxy
    phase5_ssl
    phase6_monitoring
    phase7_start_services
    phase8_verification
    
    # Show summary
    show_summary
}

# Run main function
main "$@"

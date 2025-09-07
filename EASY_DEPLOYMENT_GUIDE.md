# 🚀 Easy Deployment Guide - BrandGuard API

This guide provides simple, step-by-step instructions to deploy your BrandGuard API and models **without Docker or Kubernetes**. Perfect for quick deployment on any server or local machine.

## 📋 Prerequisites

### **System Requirements**
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 16GB+ (32GB+ recommended for production)
- **GPU**: 8GB+ VRAM (for vLLM) - Optional but recommended
- **Storage**: 20GB+ free space
- **CPU**: Multi-core processor

### **Software Requirements**
- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

## 🎯 Quick Start (5 Minutes)

### **Step 1: Clone and Setup**
```bash
# Clone the repository
git clone <your-repo-url>
cd brandReviewModels/consolidated_pipeline

# Create virtual environment
python -m venv brandguard_env
source brandguard_env/bin/activate  # On Windows: brandguard_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Start vLLM Server (Required for AI Models)**
```bash
# In a new terminal window
cd LogoDetector
python setup_vllm.py
```
*Keep this terminal open - the vLLM server needs to stay running*

### **Step 3: Start BrandGuard API**
```bash
# In another terminal window
cd consolidated_pipeline
python run.py --port 5001
```

### **Step 4: Test Your Deployment**
```bash
# Test the API
curl http://localhost:5001/api/health

# Open web interface
open http://localhost:5001
```

**🎉 That's it! Your BrandGuard API is now running!**

---

## 🔧 Detailed Deployment Options

### **Option 1: Local Development Setup**

Perfect for development and testing on your local machine.

```bash
# 1. Setup environment
cd brandReviewModels/consolidated_pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start vLLM server (Terminal 1)
cd ../LogoDetector
python setup_vllm.py

# 4. Start API server (Terminal 2)
cd ../consolidated_pipeline
python run.py --host 0.0.0.0 --port 5001 --debug
```

### **Option 2: Production Server Setup**

For deploying on a production server with better performance and reliability.

```bash
# 1. Setup system dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx supervisor

# 2. Create deployment directory
sudo mkdir -p /opt/brandguard
sudo chown $USER:$USER /opt/brandguard
cd /opt/brandguard

# 3. Clone and setup
git clone <your-repo-url> .
cd consolidated_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure production settings
cp configs/production.yaml.example configs/production.yaml
# Edit production.yaml with your settings

# 5. Start services
./deploy/start_services.sh
```

### **Option 3: Cloud Server Deployment**

For deploying on cloud providers like AWS, Google Cloud, or DigitalOcean.

```bash
# 1. Connect to your cloud server
ssh user@your-server-ip

# 2. Follow Option 2 steps above

# 3. Configure firewall
sudo ufw allow 5001
sudo ufw allow 8000

# 4. Setup SSL (optional but recommended)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 📁 Deployment Scripts

I'll create automated deployment scripts to make this even easier:

### **1. One-Click Setup Script**
```bash
# Run this single command to setup everything
./deploy/setup.sh
```

### **2. Service Management Scripts**
```bash
# Start all services
./deploy/start.sh

# Stop all services
./deploy/stop.sh

# Restart all services
./deploy/restart.sh

# Check service status
./deploy/status.sh
```

### **3. Health Monitoring**
```bash
# Check API health
./deploy/health_check.sh

# View logs
./deploy/view_logs.sh
```

---

## ⚙️ Configuration Files

### **Production Configuration**
```yaml
# configs/production.yaml
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
  file: "/var/log/brandguard/api.log"
```

### **Environment Variables**
```bash
# .env file
BRANDGUARD_HOST=0.0.0.0
BRANDGUARD_PORT=5001
BRANDGUARD_DEBUG=false
VLLM_HOST=localhost
VLLM_PORT=8000
LOG_LEVEL=INFO
```

---

## 🔍 Monitoring & Maintenance

### **Health Checks**
```bash
# API Health
curl http://localhost:5001/api/health

# vLLM Health
curl http://localhost:8000/v1/models

# System Resources
./deploy/monitor.sh
```

### **Log Management**
```bash
# View API logs
tail -f logs/api.log

# View vLLM logs
tail -f logs/vllm.log

# View all logs
./deploy/view_logs.sh
```

### **Performance Monitoring**
```bash
# Check memory usage
./deploy/check_memory.sh

# Check GPU usage (if available)
./deploy/check_gpu.sh

# Performance metrics
curl http://localhost:5001/api/metrics
```

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### **1. vLLM Server Won't Start**
```bash
# Check GPU memory
nvidia-smi

# Start with CPU fallback
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --device cpu
```

#### **2. API Connection Timeouts**
```bash
# Check if vLLM is running
curl http://localhost:8000/v1/models

# Restart vLLM server
cd LogoDetector
python setup_vllm.py
```

#### **3. Memory Issues**
```bash
# Monitor memory usage
htop

# Reduce model memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### **4. Port Conflicts**
```bash
# Check what's using the ports
lsof -i :5001
lsof -i :8000

# Kill processes if needed
sudo kill -9 <PID>
```

---

## 📊 Performance Optimization

### **For High Traffic**
```bash
# Use Gunicorn with multiple workers
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# Use Nginx as reverse proxy
sudo apt install nginx
# Configure nginx.conf (see deploy/nginx.conf)
```

### **For Better GPU Utilization**
```bash
# Set CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Use mixed precision
export VLLM_USE_MIXED_PRECISION=1
```

### **For Production Scale**
```bash
# Use systemd services
sudo cp deploy/brandguard.service /etc/systemd/system/
sudo systemctl enable brandguard
sudo systemctl start brandguard

# Setup log rotation
sudo cp deploy/logrotate.conf /etc/logrotate.d/brandguard
```

---

## 🔒 Security Considerations

### **Basic Security**
```bash
# Use environment variables for secrets
export BRANDGUARD_SECRET_KEY="your-secret-key"

# Enable HTTPS
sudo certbot --nginx -d your-domain.com

# Setup firewall
sudo ufw enable
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
```

### **API Security**
```python
# Add API key authentication
# See deploy/auth_middleware.py
```

---

## 📈 Scaling Options

### **Horizontal Scaling**
- Run multiple API instances behind a load balancer
- Use Redis for session management
- Implement API rate limiting

### **Vertical Scaling**
- Increase server RAM and CPU
- Use more powerful GPUs
- Optimize model configurations

---

## 🆘 Support & Maintenance

### **Regular Maintenance**
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Backup configurations
./deploy/backup.sh

# Clean up logs
./deploy/cleanup.sh
```

### **Getting Help**
- Check logs: `./deploy/view_logs.sh`
- Run diagnostics: `./deploy/diagnostics.sh`
- Check system resources: `./deploy/status.sh`

---

## 🎯 Next Steps

1. **Choose your deployment option** (Local/Production/Cloud)
2. **Run the setup script** for your chosen option
3. **Test the API** using the provided endpoints
4. **Configure monitoring** for production use
5. **Set up SSL** for secure connections

**Your BrandGuard API will be ready to use in minutes!** 🚀

---

*Need help? Check the troubleshooting section or run `./deploy/diagnostics.sh` for automated problem detection.*

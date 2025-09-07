# 🚀 Production Deployment - Quick Start

**Complete production deployment in one command!**

## ⚡ **One-Command Production Deploy**

```bash
# Complete production deployment
sudo ./deploy/production_deploy.sh --domain yourdomain.com
```

**That's it! Everything is handled automatically.**

## 🎯 **What This Script Does**

### **Phase 1: Basic Setup (20%)**
- ✅ Environment setup
- ✅ Dependencies installation
- ✅ Configuration setup

### **Phase 2: Security Hardening (30%)**
- ✅ Production user creation
- ✅ Firewall configuration
- ✅ Root login disabled
- ✅ User isolation

### **Phase 3: Service Management (20%)**
- ✅ Systemd service creation
- ✅ Auto-restart on reboot
- ✅ Process management

### **Phase 4: Reverse Proxy (20%)**
- ✅ Nginx installation
- ✅ Load balancing
- ✅ Rate limiting
- ✅ SSL termination

### **Phase 5: SSL/HTTPS (10%)**
- ✅ SSL certificate generation
- ✅ Auto-renewal setup
- ✅ HTTPS redirection

### **Phase 6: Monitoring (10%)**
- ✅ Log rotation
- ✅ Health checks
- ✅ Performance monitoring
- ✅ Automated alerts

## 🔧 **Command Options**

```bash
# Basic production deployment
sudo ./deploy/production_deploy.sh --domain yourdomain.com

# Skip SSL (for testing)
sudo ./deploy/production_deploy.sh --domain yourdomain.com --skip-ssl

# Skip firewall (if already configured)
sudo ./deploy/production_deploy.sh --domain yourdomain.com --skip-firewall

# Skip monitoring (minimal setup)
sudo ./deploy/production_deploy.sh --domain yourdomain.com --skip-monitoring

# Get help
./deploy/production_deploy.sh --help
```

## 📋 **Prerequisites**

### **System Requirements**
- **OS**: Ubuntu 20.04+ (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 20GB+ free space
- **CPU**: 2+ cores
- **GPU**: 8GB+ VRAM (optional but recommended)

### **Before Running**
1. **Domain name** pointing to your server
2. **Root access** to the server
3. **Repository cloned** to `/opt/brandguard/consolidated_pipeline`

## 🚀 **Step-by-Step Instructions**

### **Step 1: Prepare Your Server**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Clone your repository
sudo mkdir -p /opt/brandguard
cd /opt/brandguard
sudo git clone <your-repo-url> consolidated_pipeline
sudo chown -R $USER:$USER consolidated_pipeline
```

### **Step 2: Run Production Deployment**
```bash
# Navigate to the project
cd consolidated_pipeline

# Run production deployment
sudo ./deploy/production_deploy.sh --domain yourdomain.com
```

### **Step 3: Verify Deployment**
```bash
# Check service status
sudo systemctl status brandguard

# Check API health
curl https://yourdomain.com/health

# Check web interface
open https://yourdomain.com
```

## 📊 **What You Get**

### **✅ Complete Production Stack**
- **BrandGuard API** with all 4 models
- **vLLM Server** for AI inference
- **Nginx Reverse Proxy** with SSL
- **Systemd Service** management
- **Automated Monitoring** and health checks

### **✅ Security Features**
- **SSL/HTTPS** encryption
- **Firewall** protection
- **Rate limiting** against attacks
- **User isolation** and permissions
- **Root login** disabled

### **✅ Reliability Features**
- **Auto-restart** on failure
- **Health monitoring** with alerts
- **Log rotation** to prevent disk full
- **Service management** with systemd
- **Backup** and recovery procedures

### **✅ Performance Features**
- **Load balancing** with Nginx
- **Caching** for better performance
- **Rate limiting** for stability
- **Resource monitoring** and optimization
- **Scalable** architecture

## 🔍 **Verification Commands**

```bash
# Check all services
sudo systemctl status brandguard nginx

# Check API health
curl https://yourdomain.com/health

# Check SSL certificate
curl -I https://yourdomain.com

# View logs
sudo journalctl -u brandguard -f

# Monitor system
/opt/brandguard/monitor.sh
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Service Won't Start**
```bash
# Check service status
sudo systemctl status brandguard

# View detailed logs
sudo journalctl -u brandguard -f

# Restart service
sudo systemctl restart brandguard
```

#### **2. SSL Certificate Issues**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate
sudo certbot renew

# Check Nginx configuration
sudo nginx -t
```

#### **3. API Not Responding**
```bash
# Check if API is running
curl http://localhost:5001/api/health

# Check vLLM server
curl http://localhost:8000/v1/models

# Restart services
sudo systemctl restart brandguard
```

## 📈 **Management Commands**

| Command | Description |
|---------|-------------|
| `sudo systemctl status brandguard` | Check service status |
| `sudo systemctl restart brandguard` | Restart service |
| `sudo journalctl -u brandguard -f` | View live logs |
| `/opt/brandguard/monitor.sh` | Check system status |
| `/opt/brandguard/health_check.sh` | Run health check |
| `sudo certbot renew` | Renew SSL certificate |

## 🔒 **Security Checklist**

- ✅ **SSL/HTTPS** enabled
- ✅ **Firewall** configured
- ✅ **Root login** disabled
- ✅ **Production user** created
- ✅ **Rate limiting** enabled
- ✅ **User isolation** implemented
- ✅ **Log rotation** configured
- ✅ **Health monitoring** active

## 📊 **Performance Monitoring**

```bash
# Check system resources
htop

# Check disk usage
df -h

# Check memory usage
free -h

# Check GPU usage (if available)
nvidia-smi

# Check API performance
curl -w "@curl-format.txt" -o /dev/null -s https://yourdomain.com/api/health
```

## 🎉 **You're Done!**

Your BrandGuard API is now **production-ready** with:

- **Complete security** hardening
- **Automated monitoring** and alerts
- **SSL/HTTPS** encryption
- **Service management** with systemd
- **Load balancing** with Nginx
- **Health checks** and auto-restart
- **Log rotation** and management

**Your API is accessible at: https://yourdomain.com** 🚀

---

**Need help? Check the logs: `sudo journalctl -u brandguard -f`**


# 1. Prepare your server
sudo mkdir -p /opt/brandguard
cd /opt/brandguard
sudo git clone <your-repo-url> consolidated_pipeline

# 2. Run production deployment
cd consolidated_pipeline
sudo ./deploy/production_deploy.sh --domain yourdomain.com
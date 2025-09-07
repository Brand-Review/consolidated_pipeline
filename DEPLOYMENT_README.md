# 🚀 BrandGuard API - Easy Deployment

**Deploy your BrandGuard API in minutes without Docker or Kubernetes!**

## ⚡ Quick Start (2 Minutes)

```bash
# 1. Clone and navigate to the project
cd consolidated_pipeline

# 2. Run the one-click deployment
./deploy/quick_deploy.sh

# 3. That's it! Your API is running at http://localhost:5001
```

## 📋 What You Get

- ✅ **Complete BrandGuard API** with all 4 models
- ✅ **vLLM Server** for Qwen2.5-VL-3B-Instruct
- ✅ **Web Interface** for easy testing
- ✅ **RESTful API** for integration
- ✅ **Health Monitoring** and status checks
- ✅ **Production-ready** configuration

## 🎯 Deployment Options

### **Option 1: One-Click Deploy (Recommended)**
```bash
./deploy/quick_deploy.sh
```
*Perfect for quick testing and development*

### **Option 2: Step-by-Step Deploy**
```bash
# Setup environment
./deploy/setup.sh

# Start services
./deploy/start_services.sh

# Check status
./deploy/status.sh
```
*Perfect for production and customization*

### **Option 3: Manual Deploy**
```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start vLLM server (Terminal 1)
cd ../LogoDetector
python setup_vllm.py

# Start API server (Terminal 2)
cd ../consolidated_pipeline
python run.py --port 5001
```

## 🔧 Management Commands

| Command | Description |
|---------|-------------|
| `./deploy/quick_deploy.sh` | One-click deployment |
| `./deploy/start_services.sh` | Start all services |
| `./deploy/stop_services.sh` | Stop all services |
| `./deploy/status.sh` | Check service status |
| `./deploy/setup.sh` | Setup environment only |

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/analyze` | POST | Full brand analysis |
| `/api/analyze/color` | POST | Color analysis only |
| `/api/analyze/typography` | POST | Typography analysis only |
| `/api/analyze/copywriting` | POST | Copywriting analysis only |
| `/api/analyze/logo` | POST | Logo detection only |

## 📊 Service Status

```bash
# Check if everything is running
./deploy/status.sh

# Test API health
curl http://localhost:5001/api/health

# Test vLLM server
curl http://localhost:8000/v1/models
```

## 🔍 Testing Your Deployment

### **1. Web Interface Test**
Open http://localhost:5001 in your browser

### **2. API Test**
```bash
# Health check
curl http://localhost:5001/api/health

# Upload an image for analysis
curl -X POST http://localhost:5001/api/analyze \
  -F "file=@your_image.jpg" \
  -F "enable_color=true" \
  -F "enable_logo=true"
```

### **3. Python Test**
```python
import requests

# Test API
response = requests.get("http://localhost:5001/api/health")
print(response.json())

# Test analysis
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5001/api/analyze",
        files={"file": f},
        data={"enable_color": "true", "enable_logo": "true"}
    )
print(response.json())
```

## 📁 Project Structure

```
consolidated_pipeline/
├── deploy/                 # Deployment scripts
│   ├── quick_deploy.sh    # One-click deployment
│   ├── setup.sh           # Environment setup
│   ├── start_services.sh  # Start all services
│   ├── stop_services.sh   # Stop all services
│   └── status.sh          # Check status
├── configs/               # Configuration files
│   └── production.yaml    # Production settings
├── logs/                  # Log files
├── uploads/               # Upload directory
├── results/               # Analysis results
├── src/                   # Source code
├── app.py                 # Main API application
├── run.py                 # Server runner
└── requirements.txt       # Dependencies
```

## ⚙️ Configuration

### **Production Settings**
Edit `configs/production.yaml` to customize:
- Server host/port
- Model parameters
- Logging levels
- File upload limits
- Security settings

### **Environment Variables**
Create `.env` file:
```bash
BRANDGUARD_HOST=0.0.0.0
BRANDGUARD_PORT=5001
VLLM_HOST=localhost
VLLM_PORT=8000
LOG_LEVEL=INFO
```

## 🚨 Troubleshooting

### **Common Issues**

#### **Services Won't Start**
```bash
# Check if ports are in use
lsof -i :5001
lsof -i :8000

# Kill processes if needed
sudo kill -9 <PID>
```

#### **vLLM Server Issues**
```bash
# Check GPU memory
nvidia-smi

# Start with CPU fallback
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --device cpu
```

#### **Memory Issues**
```bash
# Check memory usage
htop

# Reduce model memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **Get Help**
```bash
# Run diagnostics
./deploy/status.sh

# View logs
tail -f logs/api.log
tail -f logs/vllm.log

# Check system resources
htop
nvidia-smi
```

## 📈 Performance Tips

### **For Better Performance**
- Use GPU for vLLM server
- Increase server RAM (16GB+ recommended)
- Use SSD storage
- Enable caching in production config

### **For High Traffic**
- Use multiple API workers
- Implement load balancing
- Use Redis for session management
- Enable rate limiting

## 🔒 Security

### **Basic Security**
- Use HTTPS in production
- Set up firewall rules
- Use environment variables for secrets
- Enable API authentication

### **Production Security**
- Use reverse proxy (Nginx)
- Enable SSL/TLS
- Implement API rate limiting
- Regular security updates

## 📊 Monitoring

### **Health Checks**
```bash
# API health
curl http://localhost:5001/api/health

# vLLM health
curl http://localhost:8000/v1/models

# System resources
htop
nvidia-smi
```

### **Logs**
```bash
# View all logs
tail -f logs/*.log

# View specific logs
tail -f logs/api.log
tail -f logs/vllm.log
```

## 🆘 Support

### **Quick Help**
1. Check service status: `./deploy/status.sh`
2. View logs: `tail -f logs/api.log`
3. Restart services: `./deploy/stop_services.sh && ./deploy/start_services.sh`

### **Common Commands**
```bash
# Full restart
./deploy/stop_services.sh
./deploy/start_services.sh

# Check everything
./deploy/status.sh

# View logs
tail -f logs/api.log logs/vllm.log
```

---

## 🎉 You're Ready!

Your BrandGuard API is now deployed and ready to use! 

- **Web Interface**: http://localhost:5001
- **API Documentation**: http://localhost:5001/api/docs
- **Health Check**: http://localhost:5001/api/health

**Happy analyzing!** 🚀

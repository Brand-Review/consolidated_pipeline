# BrandGuard AI Pipeline - Docker Deployment Guide

This guide provides comprehensive instructions for deploying the BrandGuard AI Pipeline using Docker. The pipeline automatically clones and integrates all model repositories from GitHub.

## 🏗️ Architecture Overview

The Docker setup includes:
- **Main API Service**: Flask-based consolidated pipeline
- **Model Integration**: Automatic cloning of 4 model repositories
- **Reverse Proxy**: Nginx for production deployment
- **Monitoring**: Prometheus and Grafana (development mode)
- **Caching**: Redis for performance optimization

## 📋 Prerequisites

- Docker 20.10+ 
- Docker Compose 2.0+
- Git (for local development)
- 8GB+ RAM recommended
- 20GB+ disk space for models

## 🚀 Quick Start

### Option 1: One-Command Deployment
```bash
# Clone the repository
git clone <your-consolidated-pipeline-repo>
cd consolidated_pipeline

# Run quick start
./deploy/quick_start.sh
```

### Option 2: Manual Deployment
```bash
# Build and start services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔧 Configuration

### Environment Variables

The pipeline uses environment variables for configuration. Copy `docker.env` to `.env` and modify as needed:

```bash
cp docker.env .env
```

Key configuration options:
- `MAX_FILE_SIZE`: Maximum file upload size (default: 50MB)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Deployment environment (development, production)

### Model Configuration

Each model can be configured through YAML files in the `configs/` directory:
- `color_palette.yaml`: Color analysis settings
- `typography_rules.yaml`: Font and typography rules
- `brand_voice.yaml`: Copywriting tone settings
- `logo_detection.yaml`: Logo detection parameters

## 📁 Directory Structure

```
consolidated_pipeline/
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml        # Production services
├── docker-compose.override.yml # Development overrides
├── docker.env               # Environment configuration
├── nginx.conf               # Reverse proxy configuration
├── setup_models.sh          # Model setup script
├── deploy/
│   ├── docker_deploy.sh     # Full deployment script
│   └── quick_start.sh       # Quick start script
├── monitoring/
│   └── prometheus.yml       # Monitoring configuration
├── configs/                 # Model configurations
├── src/brandguard/          # Source code
├── uploads/                 # File uploads (mounted volume)
├── results/                 # Analysis results (mounted volume)
└── models/                  # Model files (mounted volume)
```

## 🔄 Model Integration

The pipeline automatically clones and integrates these repositories:

1. **Font Typography Checker**
   - Repository: `FontIdentification-PaddleOCR-gaborcselle_font_identifier`
   - Purpose: Font identification and typography analysis

2. **Copywriting Tone Checker**
   - Repository: `BrandVoiceChecker-PaddleOCR-vLLM-Qwen2.5_3B`
   - Purpose: Brand voice and tone analysis

3. **Color Palette Checker**
   - Repository: `ColorExtraction-with-Kmeans-ColorSpaceProcessing-CIEDE2000DistanceColorMatching`
   - Purpose: Color extraction and palette analysis

4. **Logo Detection**
   - Repository: `LogoDetectionModels-With-BrandPlacementRulesEngine-and-ValidationPipeline`
   - Purpose: Logo detection and placement validation

## 🐳 Docker Services

### Production Services
- `brandguard-api`: Main FastAPI application
- `nginx`: Reverse proxy and load balancer
- `redis`: Caching and session storage

### Development Services
- `postgres`: Database for development
- `prometheus`: Metrics collection
- `grafana`: Monitoring dashboard

## 🚀 Deployment Modes

### Development Mode
```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Production Mode
```bash
# Start production services
docker-compose up -d

# Access services
# API: http://localhost:8000
# Nginx: http://localhost (port 80)
```

## 📊 Monitoring and Logs

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f brandguard-api

# Last 100 lines
docker-compose logs --tail=100 brandguard-api
```

### Health Checks
```bash
# API health
curl http://localhost:8000/api/health

# Service status
docker-compose ps
```

### Monitoring (Development Mode)
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
- **Prometheus**: http://localhost:9090

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Out of Memory**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop -> Settings -> Resources -> Memory
   ```

3. **Model Download Failures**
   ```bash
   # Rebuild with no cache
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   chmod -R 755 .
   ```

### Debug Mode
```bash
# Run in debug mode
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Access container shell
docker-compose exec brandguard-api bash
```

## 🔄 Updates and Maintenance

### Update Models
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Backup Data
```bash
# Backup volumes
docker run --rm -v brandguard_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads.tar.gz -C /data .
docker run --rm -v brandguard_results:/data -v $(pwd):/backup alpine tar czf /backup/results.tar.gz -C /data .
```

### Clean Up
```bash
# Remove containers and volumes
docker-compose down --volumes --remove-orphans

# Clean up images
docker system prune -a
```

## 🌐 API Endpoints

### Main Endpoints
- `GET /`: Web interface
- `GET /api/health`: Health check
- `GET /api/config`: Configuration info
- `POST /api/analyze`: Full analysis pipeline

### Individual Model Endpoints
- `POST /api/analyze/color`: Color analysis only
- `POST /api/analyze/typography`: Typography analysis only
- `POST /api/analyze/copywriting`: Copywriting analysis only
- `POST /api/analyze/logo`: Logo detection only

### Documentation
- `GET /`: Main web interface for file uploads and analysis

## 🔒 Security Considerations

1. **Change Default Passwords**: Update default passwords in production
2. **Use HTTPS**: Configure SSL certificates for production
3. **Network Security**: Use proper firewall rules
4. **Secrets Management**: Use Docker secrets for sensitive data
5. **Regular Updates**: Keep base images and dependencies updated

## 📈 Performance Optimization

1. **Resource Limits**: Set appropriate CPU and memory limits
2. **Caching**: Enable Redis caching for better performance
3. **Load Balancing**: Use multiple API instances behind Nginx
4. **Model Optimization**: Use quantized models for faster inference
5. **CDN**: Use CDN for static assets

## 🆘 Support

For issues and support:
1. Check the logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Test individual services: `docker-compose exec brandguard-api python -c "import src.brandguard"`
4. Check resource usage: `docker stats`

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

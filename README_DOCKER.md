# 🐳 BrandGuard AI Pipeline - Docker Quick Start

> **One-command deployment** for the complete BrandGuard AI Pipeline with automatic model integration.

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone the repository
git clone <your-consolidated-pipeline-repo>
cd consolidated_pipeline

# 2. Deploy everything
./deploy/quick_start.sh

# 3. Access your API
open http://localhost:8000
```

That's it! 🎉

## 📋 What Gets Deployed

✅ **4 AI Models** automatically cloned and integrated:
- 🎨 Color Palette Analysis
- 🔤 Font & Typography Detection  
- ✍️ Brand Voice & Copywriting Analysis
- 🏷️ Logo Detection & Placement Validation

✅ **Production-Ready Stack**:
- Flask backend with web interface
- Nginx reverse proxy
- Redis caching
- Health monitoring
- File upload handling

✅ **Development Tools** (optional):
- Grafana dashboards
- Prometheus metrics
- PostgreSQL database

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Main web interface |
| **Health** | http://localhost:8000/api/health | Service health check |
| **Grafana** | http://localhost:3000 | Monitoring (dev mode) |
| **Prometheus** | http://localhost:9090 | Metrics (dev mode) |

## 🔧 Management Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Check status
docker-compose ps

# Update models
git pull && docker-compose up --build -d
```

## 📊 API Usage Examples

### Full Analysis Pipeline
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@your-image.jpg" \
  -F "enable_color=true" \
  -F "enable_typography=true" \
  -F "enable_copywriting=true" \
  -F "enable_logo=true"
```

### Individual Model Analysis
```bash
# Color analysis only
curl -X POST "http://localhost:8000/api/analyze/color" \
  -F "file=@your-image.jpg"

# Typography analysis only  
curl -X POST "http://localhost:8000/api/analyze/typography" \
  -F "file=@your-image.jpg"

# Copywriting analysis only
curl -X POST "http://localhost:8000/api/analyze/copywriting" \
  -F "file=@your-image.jpg"

# Logo detection only
curl -X POST "http://localhost:8000/api/analyze/logo" \
  -F "file=@your-image.jpg"
```

## 🛠️ Configuration

### Environment Variables
Copy and modify the environment file:
```bash
cp docker.env .env
# Edit .env with your settings
```

### Model Settings
Edit YAML files in `configs/` directory:
- `color_palette.yaml` - Color analysis settings
- `typography_rules.yaml` - Font detection rules  
- `brand_voice.yaml` - Copywriting tone settings
- `logo_detection.yaml` - Logo detection parameters

## 🔍 Troubleshooting

### Common Issues

**Port already in use:**
```bash
docker-compose down
# Or change ports in docker-compose.yml
```

**Out of memory:**
```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB+ RAM
```

**Models not loading:**
```bash
# Rebuild with no cache
docker-compose build --no-cache
docker-compose up -d
```

**Permission issues:**
```bash
sudo chown -R $USER:$USER .
chmod -R 755 .
```

### Debug Mode
```bash
# Run with debug output
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Access container shell
docker-compose exec brandguard-api bash
```

## 📁 Project Structure

```
consolidated_pipeline/
├── 🐳 Dockerfile              # Multi-stage build
├── 🐳 docker-compose.yml      # Production services  
├── 🐳 docker-compose.override.yml # Development extras
├── 📄 docker.env              # Environment config
├── 🚀 deploy/
│   ├── quick_start.sh         # One-command deploy
│   └── docker_deploy.sh       # Full deployment
├── 📊 monitoring/             # Prometheus config
├── ⚙️ configs/               # Model configurations
├── 📁 src/brandguard/         # Source code
├── 📁 uploads/               # File uploads (volume)
├── 📁 results/               # Analysis results (volume)
└── 📁 models/                # Model files (volume)
```

## 🔄 Model Integration

The pipeline automatically clones these repositories:

| Model | Repository | Purpose |
|-------|------------|---------|
| 🎨 Color | `ColorExtraction-with-Kmeans-ColorSpaceProcessing-CIEDE2000DistanceColorMatching` | Color palette analysis |
| 🔤 Typography | `FontIdentification-PaddleOCR-gaborcselle_font_identifier` | Font detection |
| ✍️ Copywriting | `BrandVoiceChecker-PaddleOCR-vLLM-Qwen2.5_3B` | Brand voice analysis |
| 🏷️ Logo | `LogoDetectionModels-With-BrandPlacementRulesEngine-and-ValidationPipeline` | Logo detection |

## 🚀 Production Deployment

For production deployment, see the [Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md) for:
- SSL/HTTPS configuration
- Load balancing setup
- Security hardening
- Monitoring and alerting
- Backup strategies

## 📞 Support

- 📖 **Full Documentation**: [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)
- 🐛 **Issues**: Check logs with `docker-compose logs -f`
- 🔧 **Debug**: Run in debug mode for detailed output
- 📊 **Monitoring**: Use Grafana dashboard in development mode

---

**Ready to analyze your brand assets?** Run `./deploy/quick_start.sh` and start building! 🚀

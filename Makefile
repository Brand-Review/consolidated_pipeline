# BrandGuard AI Pipeline - Makefile
# Convenient commands for Docker deployment and management

.PHONY: help build up down restart logs status clean dev prod test health

# Default target
help:
	@echo "BrandGuard AI Pipeline - Available Commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make quick     - One-command deployment"
	@echo "  make dev       - Start development environment"
	@echo "  make prod      - Start production environment"
	@echo ""
	@echo "🔧 Management:"
	@echo "  make build     - Build Docker images"
	@echo "  make up        - Start services"
	@echo "  make down      - Stop services"
	@echo "  make restart   - Restart services"
	@echo "  make logs      - View logs"
	@echo "  make status    - Check service status"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean     - Clean up containers and volumes"
	@echo "  make test      - Run health checks"
	@echo "  make health    - Check API health"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  make monitor   - Start with monitoring (dev mode)"
	@echo "  make grafana   - Open Grafana dashboard"
	@echo "  make prometheus - Open Prometheus metrics"

# Quick start
quick:
	@echo "🚀 Starting BrandGuard AI Pipeline..."
	@./deploy/quick_start.sh

# Development environment
dev:
	@echo "🔧 Starting development environment..."
	@docker-compose -f docker-compose.yml -f docker-compose.override.yml up --build -d
	@echo "✅ Development environment started!"
	@echo "🌐 API: http://localhost:8000"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"

# Production environment
prod:
	@echo "🏭 Starting production environment..."
	@docker-compose up --build -d
	@echo "✅ Production environment started!"
	@echo "🌐 API: http://localhost:8000"

# Build images
build:
	@echo "🔨 Building Docker images..."
	@docker-compose build --no-cache

# Start services
up:
	@echo "⬆️  Starting services..."
	@docker-compose up -d

# Stop services
down:
	@echo "⬇️  Stopping services..."
	@docker-compose down

# Restart services
restart:
	@echo "🔄 Restarting services..."
	@docker-compose restart

# View logs
logs:
	@echo "📋 Viewing logs..."
	@docker-compose logs -f

# Check status
status:
	@echo "📊 Service status:"
	@docker-compose ps
	@echo ""
	@echo "🔍 Container health:"
	@docker stats --no-stream

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down --volumes --remove-orphans
	@docker system prune -f
	@echo "✅ Cleanup completed"

# Run tests
test:
	@echo "🧪 Running health checks..."
	@curl -f http://localhost:8000/api/health || echo "❌ API not responding"
	@echo "✅ Health checks completed"

# Check API health
health:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8000/api/health | jq . || echo "❌ API not responding"

# Start with monitoring
monitor: dev
	@echo "📊 Monitoring services started!"
	@echo "🌐 Grafana: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"

# Open Grafana
grafana:
	@echo "📊 Opening Grafana dashboard..."
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Please open http://localhost:3000"

# Open Prometheus
prometheus:
	@echo "📈 Opening Prometheus metrics..."
	@open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Please open http://localhost:9090"

# Setup models (manual)
setup-models:
	@echo "🔧 Setting up models..."
	@./setup_models.sh

# Update models
update:
	@echo "🔄 Updating models..."
	@git pull origin main
	@docker-compose down
	@docker-compose up --build -d
	@echo "✅ Models updated!"

# Backup data
backup:
	@echo "💾 Backing up data..."
	@mkdir -p backups
	@docker run --rm -v brandguard_uploads:/data -v $(PWD)/backups:/backup alpine tar czf /backup/uploads-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@docker run --rm -v brandguard_results:/data -v $(PWD)/backups:/backup alpine tar czf /backup/results-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "✅ Backup completed in backups/ directory"

# Restore data
restore:
	@echo "📥 Restoring data..."
	@echo "Available backups:"
	@ls -la backups/
	@echo "Please specify backup file: make restore BACKUP=filename.tar.gz"
	@if [ -n "$(BACKUP)" ]; then \
		docker run --rm -v brandguard_uploads:/data -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP) -C /data; \
		echo "✅ Data restored from $(BACKUP)"; \
	fi

# Show logs for specific service
logs-api:
	@docker-compose logs -f brandguard-api

logs-nginx:
	@docker-compose logs -f nginx

logs-redis:
	@docker-compose logs -f redis

# Execute command in container
exec:
	@docker-compose exec brandguard-api bash

# Show resource usage
stats:
	@docker stats

# Show disk usage
disk:
	@docker system df

# Show network info
network:
	@docker network ls
	@docker network inspect brandguard_brandguard-network

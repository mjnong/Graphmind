# Makefile for Custom GraphRAG Project
# A comprehensive, user-friendly interface to manage your file upload and processing system

# Colors for better UX
GREEN=\033[0;32m
YELLOW=\033[0;33m
RED=\033[0;31m
BLUE=\033[0;34m
PURPLE=\033[0;35m
CYAN=\033[0;36m
WHITE=\033[1;37m
NC=\033[0m # No Color

# Status indicators
INFO=$(BLUE)ℹ$(NC)
SUCCESS=$(GREEN)✓$(NC)
ERROR=$(RED)✗$(NC)
WARNING=$(YELLOW)⚠$(NC)
ROCKET=$(PURPLE)🚀$(NC)
GEAR=$(CYAN)⚙$(NC)
DOCS=$(WHITE)📖$(NC)

# Project settings
DOCKER_COMPOSE=docker compose -f Docker/docker-compose.yaml
PROJECT_NAME=custom-graphRAG
API_URL=http://localhost:8888
WS_URL=ws://localhost:8888/files/ws

.DEFAULT_GOAL := help

# ==========================================
# HELP & INFORMATION
# ==========================================

help: ## 📖 Show this help message
	@echo "$(WHITE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(WHITE)║                    $(PURPLE)Custom GraphRAG$(WHITE) - File Upload System          ║$(NC)"
	@echo "$(WHITE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(DOCS) $(WHITE)Quick Start:$(NC)"
	@echo "  $(CYAN)make setup$(NC)        - 🎯 Complete project setup (recommended for first time)"
	@echo "  $(CYAN)make dev$(NC)          - 🚀 Start development environment"
	@echo "  $(CYAN)make test-upload$(NC)  - 🧪 Test file upload system"
	@echo ""
	@echo "$(DOCS) $(WHITE)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-16s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(dev|build|run|stop|logs|clean)"
	@echo ""
	@echo "$(DOCS) $(WHITE)Database & Migrations:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-16s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(db-|migrate)"
	@echo ""
	@echo "$(DOCS) $(WHITE)File Processing & Workers:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-16s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(worker|file|upload)"
	@echo ""
	@echo "$(DOCS) $(WHITE)Monitoring & Debugging:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-16s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(status|health|debug|monitor)"
	@echo ""
	@echo "$(DOCS) $(WHITE)Maintenance:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-16s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(clean|reset|backup)"
	@echo ""
	@echo "$(INFO) $(WHITE)Service URLs:$(NC)"
	@echo "  API:       $(GREEN)$(API_URL)$(NC)"
	@echo "  WebSocket: $(GREEN)$(WS_URL)$(NC)"
	@echo "  PostgreSQL: $(GREEN)localhost:8432$(NC)"
	@echo "  Neo4j:     $(GREEN)localhost:8474$(NC) (HTTP) / $(GREEN)localhost:8687$(NC) (Bolt)"
	@echo "  Redis:     $(GREEN)localhost:6379$(NC)"

status: ## 📊 Show comprehensive system status
	@echo "$(INFO) $(WHITE)System Status Check$(NC)"
	@echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"
	@$(MAKE) --no-print-directory _check_services
	@echo ""
	@$(MAKE) --no-print-directory _check_api_health
	@echo ""
	@$(MAKE) --no-print-directory _check_database_health
	@echo ""
	@$(MAKE) --no-print-directory _check_uploads

# ==========================================
# QUICK START & SETUP
# ==========================================

setup: ## 🎯 Complete first-time project setup
	@echo "$(ROCKET) $(WHITE)Setting up Custom GraphRAG Project$(NC)"
	@echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(INFO) Step 1: Creating required directories..."
	@mkdir -p uploads temp_processing history
	@echo "$(INFO) Step 2: Building containers..."
	@$(MAKE) --no-print-directory build
	@echo ""
	@echo "$(INFO) Step 3: Starting services..."
	@$(MAKE) --no-print-directory start
	@echo ""
	@echo "$(INFO) Step 4: Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "$(INFO) Step 5: Running database migrations..."
	@$(MAKE) --no-print-directory migrate
	@echo ""
	@echo "$(INFO) Step 6: Final health check..."
	@$(MAKE) --no-print-directory status
	@echo ""
	@echo "$(SUCCESS) $(WHITE)Setup complete! Your file upload system is ready.$(NC)"
	@echo "$(INFO) Try: $(CYAN)make test-upload$(NC) to test the system"
	@echo "$(INFO) Or:  $(CYAN)make dev$(NC) to start development mode"

dev: start ## 🚀 Start development environment with hot reload
	@echo "$(ROCKET) $(WHITE)Development mode started!$(NC)"
	@echo "$(INFO) Services are running with hot reload enabled"
	@echo "$(INFO) Open $(GREEN)upload_test.html$(NC) in your browser to test uploads"
	@echo "$(WARNING) Press $(CYAN)Ctrl+C$(NC) to stop, or use $(CYAN)make stop$(NC)"

# ==========================================
# DEVELOPMENT WORKFLOW COMMANDS
# ==========================================

env-setup: ## 🔧 Set up local development environment for migrations
	@echo "$(GEAR) Setting up local development environment..."
	@if ! command -v python3 >/dev/null 2>&1; then \
		echo "$(ERROR) Python 3 is required but not installed"; \
		exit 1; \
	fi
	@echo "$(INFO) Installing required packages for migrations..."
	@pip install alembic psycopg2-binary python-dotenv pydantic pydantic-settings sqlalchemy
	@echo "$(SUCCESS) Local environment setup complete"
	@echo "$(INFO) You can now run migrations with: $(CYAN)make migrate$(NC)"

start-services: ## 🚀 Start only the infrastructure services (DB, Redis, Neo4j)
	@echo "$(GEAR) Starting infrastructure services..."
	@$(DOCKER_COMPOSE) up -d db broker neo4j && \
		echo "$(SUCCESS) Infrastructure services started" || \
		(echo "$(ERROR) Failed to start services" && exit 1)
	@echo "$(INFO) Waiting for services to be ready..."
	@sleep 5
	@$(MAKE) --no-print-directory _wait_for_infrastructure

dev-full: start-services ## 🚀 Complete dev workflow: infrastructure + migrations + app services  
	@echo "$(INFO) Running migrations..."
	@$(MAKE) --no-print-directory migrate
	@echo "$(INFO) Starting application services..."
	@$(DOCKER_COMPOSE) up -d graphrag worker
	@echo "$(SUCCESS) Full development environment is ready!"
	@echo "$(INFO) API available at: $(GREEN)$(API_URL)$(NC)"

dev-local: start-services ## 🏠 Development with local API (services in Docker, API on host)
	@echo "$(INFO) Infrastructure services started"
	@echo "$(INFO) Run migrations: $(CYAN)make migrate$(NC)"
	@echo "$(INFO) Then start your local API server with uvicorn or your IDE"
	@echo "$(INFO) Database available at: $(GREEN)localhost:8432$(NC)"

mcp-inspect: ## 🔍 Run MCP inspector for debugging
	@echo "$(INFO) Starting MCP inspector..."
	@echo "$(WARNING) Make sure the service is running first with 'make start'"
	@npx @modelcontextprotocol/inspector
	
# ==========================================
# CORE DOCKER OPERATIONS
# ==========================================

build: ## 🔨 Build all Docker containers
	@echo "$(GEAR) Building Docker containers..."
	@$(DOCKER_COMPOSE) build --parallel && \
		echo "$(SUCCESS) Containers built successfully" || \
		(echo "$(ERROR) Failed to build containers" && exit 1)

start: ## ▶️  Start all services
	@echo "$(GEAR) Starting all services..."
	@$(DOCKER_COMPOSE) up -d && \
		echo "$(SUCCESS) All services started" || \
		(echo "$(ERROR) Failed to start services" && exit 1)
	@echo "$(INFO) Waiting for services to initialize..."
	@sleep 5
	@$(MAKE) --no-print-directory _wait_for_services

stop: ## ⏹️  Stop all services
	@echo "$(INFO) Stopping all services..."
	@$(DOCKER_COMPOSE) down && \
		echo "$(SUCCESS) All services stopped" || \
		echo "$(WARNING) Some services were not running"

restart: stop start ## 🔄 Restart all services

logs: ## 📋 View logs from all services
	@echo "$(INFO) Viewing logs (press Ctrl+C to exit)..."
	@$(DOCKER_COMPOSE) logs -f

logs-api: ## 📋 View API service logs only
	@$(DOCKER_COMPOSE) logs -f graphrag

logs-worker: ## 📋 View worker logs only
	@$(DOCKER_COMPOSE) logs -f worker

logs-db: ## 📋 View database logs only
	@$(DOCKER_COMPOSE) logs -f db

# ==========================================
# DATABASE OPERATIONS
# ==========================================

db-shell: ## 🐘 Open PostgreSQL shell
	@echo "$(INFO) Opening PostgreSQL shell..."
	@$(DOCKER_COMPOSE) exec db psql -U postgres -d postgres

migrate: ## 🔄 Run database migrations (from host)
	@echo "$(GEAR) Running database migrations from host..."
	@if ! command -v alembic >/dev/null 2>&1; then \
		echo "$(ERROR) Alembic not found. Please install with: pip install alembic"; \
		exit 1; \
	fi
	@if [ ! -f "alembic.ini" ]; then \
		echo "$(ERROR) alembic.ini not found in current directory"; \
		exit 1; \
	fi
	@alembic upgrade head && \
		echo "$(SUCCESS) Migrations completed successfully" || \
		(echo "$(ERROR) Migration failed" && exit 1)

migrate-create: ## 📝 Create new migration (use: make migrate-create msg="your message")
	@if [ -z "$(msg)" ]; then \
		echo "$(ERROR) Migration message required. Use: make migrate-create msg=\"your message\""; \
		exit 1; \
	fi
	@if ! command -v alembic >/dev/null 2>&1; then \
		echo "$(ERROR) Alembic not found. Please install with: pip install alembic"; \
		exit 1; \
	fi
	@echo "$(GEAR) Creating migration: $(msg)"
	@alembic revision --autogenerate -m "$(msg)" && \
		echo "$(SUCCESS) Migration created: $(GREEN)$(msg)$(NC)" || \
		(echo "$(ERROR) Failed to create migration" && exit 1)
	@echo "$(INFO) Review the migration file in $(GREEN)alembic/versions/$(NC) before applying"

migrate-history: ## 📜 Show migration history
	@if ! command -v alembic >/dev/null 2>&1; then \
		echo "$(ERROR) Alembic not found. Please install with: pip install alembic"; \
		exit 1; \
	fi
	@alembic history --verbose

migrate-current: ## 📍 Show current migration status
	@if ! command -v alembic >/dev/null 2>&1; then \
		echo "$(ERROR) Alembic not found. Please install with: pip install alembic"; \
		exit 1; \
	fi
	@alembic current

migrate-rollback: ## ⏪ Rollback last migration (with confirmation)
	@if ! command -v alembic >/dev/null 2>&1; then \
		echo "$(ERROR) Alembic not found. Please install with: pip install alembic"; \
		exit 1; \
	fi
	@echo "$(WARNING) This will rollback the database to the previous migration!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(INFO) Rolling back..."; \
		alembic downgrade -1 && \
		echo "$(SUCCESS) Rollback complete"; \
	else \
		echo "$(INFO) Rollback cancelled"; \
	fi

db-reset: ## 💣 Reset database (WARNING: destroys all data)
	@echo "$(ERROR) $(WHITE)WARNING: This will destroy ALL database data!$(NC)"
	@read -p "Type 'DELETE' to confirm: " -r; \
	if [[ $$REPLY == "DELETE" ]]; then \
		echo "$(INFO) Resetting database..."; \
		$(DOCKER_COMPOSE) down -v db && \
		$(DOCKER_COMPOSE) up -d db && \
		sleep 10 && \
		$(MAKE) --no-print-directory migrate && \
		echo "$(SUCCESS) Database reset complete"; \
	else \
		echo "$(INFO) Database reset cancelled"; \
	fi

# ==========================================
# FILE UPLOAD & PROCESSING
# ==========================================

test-upload: ## 🧪 Test file upload system
	@echo "$(ROCKET) $(WHITE)Testing File Upload System$(NC)"
	@echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"
	@if command -v python3 >/dev/null 2>&1; then \
		echo "$(INFO) Opening upload test page..."; \
		python3 -m http.server 8080 --directory . >/dev/null 2>&1 & \
		sleep 2; \
		echo "$(SUCCESS) Test server started on $(GREEN)http://localhost:8080/upload_test.html$(NC)"; \
		echo "$(INFO) Upload test page should open in your browser"; \
		if command -v open >/dev/null 2>&1; then \
			open http://localhost:8080/upload_test.html; \
		elif command -v xdg-open >/dev/null 2>&1; then \
			xdg-open http://localhost:8080/upload_test.html; \
		else \
			echo "$(INFO) Manually open: $(GREEN)http://localhost:8080/upload_test.html$(NC)"; \
		fi; \
		echo "$(WARNING) Press Enter when done testing to stop the test server..."; \
		read; \
		pkill -f "python3 -m http.server 8080" 2>/dev/null || true; \
		echo "$(SUCCESS) Test server stopped"; \
	else \
		echo "$(ERROR) Python3 not found. Manually open upload_test.html in your browser"; \
		echo "$(INFO) Make sure API is running: $(GREEN)$(API_URL)$(NC)"; \
	fi

worker-status: ## 👷 Check worker status and queue
	@echo "$(INFO) Worker Status:"
	@$(DOCKER_COMPOSE) exec worker celery -A src.app.celery.worker inspect active || \
		echo "$(WARNING) No active workers found"

worker-restart: ## 🔄 Restart worker service
	@echo "$(INFO) Restarting worker..."
	@$(DOCKER_COMPOSE) restart worker && \
		echo "$(SUCCESS) Worker restarted"

worker-shell: ## 💻 Open worker shell for debugging
	@$(DOCKER_COMPOSE) exec worker bash

file-stats: ## 📊 Show file upload statistics
	@echo "$(INFO) File Upload Statistics:"
	@curl -s $(API_URL)/files/uploads | python3 -m json.tool 2>/dev/null || \
		echo "$(WARNING) API not accessible or no uploads found"

# ==========================================
# MONITORING & HEALTH CHECKS
# ==========================================

health: ## 🏥 Comprehensive health check
	@echo "$(INFO) $(WHITE)Health Check Results$(NC)"
	@echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"
	@$(MAKE) --no-print-directory _check_api_health
	@$(MAKE) --no-print-directory _check_database_health
	@$(MAKE) --no-print-directory _check_worker_health
	@$(MAKE) --no-print-directory _check_storage_health

debug: ## 🐛 Debug mode - show detailed service information
	@echo "$(INFO) $(WHITE)Debug Information$(NC)"
	@echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(INFO) Container Status:"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(INFO) Recent Logs (last 50 lines):"
	@$(DOCKER_COMPOSE) logs --tail=50
	@echo ""
	@echo "$(INFO) System Resources:"
	@docker stats --no-stream $(shell $(DOCKER_COMPOSE) ps -q) 2>/dev/null || echo "No containers running"

monitor: ## 📊 Real-time monitoring dashboard
	@echo "$(INFO) Starting real-time monitoring..."
	@echo "$(INFO) Press Ctrl+C to stop monitoring"
	@while true; do \
		clear; \
		echo "$(WHITE)Custom GraphRAG - Real-time Monitor$(NC)"; \
		echo "$(WHITE)════════════════════════════════════════════════════════════════$(NC)"; \
		echo "Updated: $$(date)"; \
		echo ""; \
		$(MAKE) --no-print-directory _check_services 2>/dev/null; \
		echo ""; \
		echo "$(INFO) Container Stats:"; \
		docker stats --no-stream $(shell $(DOCKER_COMPOSE) ps -q) 2>/dev/null | head -6 || echo "No containers running"; \
		sleep 5; \
	done

# ==========================================
# MAINTENANCE & CLEANUP
# ==========================================

clean: ## 🧹 Clean up Docker resources
	@echo "$(INFO) Cleaning up Docker resources..."
	@docker system prune -f && \
		echo "$(SUCCESS) Cleanup complete"

clean-all: ## 💣 Deep clean (removes all containers, images, volumes)
	@echo "$(WARNING) This will remove ALL Docker data for this project!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(INFO) Performing deep clean..."; \
		$(DOCKER_COMPOSE) down -v --rmi all && \
		docker system prune -af && \
		echo "$(SUCCESS) Deep clean complete"; \
	else \
		echo "$(INFO) Deep clean cancelled"; \
	fi

backup-db: ## 💾 Backup database to file
	@echo "$(INFO) Creating database backup..."
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec -T db pg_dump -U postgres postgres > backups/backup_$$(date +%Y%m%d_%H%M%S).sql && \
		echo "$(SUCCESS) Database backup created in backups/ directory"

restore-db: ## 📥 Restore database from backup (specify file with: make restore-db file=backup.sql)
	@if [ -z "$(file)" ]; then \
		echo "$(ERROR) Backup file required. Use: make restore-db file=backup.sql"; \
		exit 1; \
	fi
	@if [ ! -f "$(file)" ]; then \
		echo "$(ERROR) Backup file $(file) not found"; \
		exit 1; \
	fi
	@echo "$(WARNING) This will overwrite the current database!"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(INFO) Restoring database..."; \
		$(DOCKER_COMPOSE) exec -T db psql -U postgres postgres < $(file) && \
		echo "$(SUCCESS) Database restored from $(file)"; \
	else \
		echo "$(INFO) Restore cancelled"; \
	fi

uv-update-lock: ## 🔄 Update uv lock file with latest dependencies
	@echo "$(INFO) Updating Python dependencies with uv.."
	@uv lock --upgrade --prerelease=allow && echo "$(SUCCESS) Dependencies updated successfully"
	@uv sync --upgrade --prerelease=allow && echo "$(SUCCESS) Dependencies synchronized successfully"
	@echo "$(SUCCESS) Lock file updated. Please commit changes to 'uv.lock' if necessary."

# ==========================================
# INTERNAL HELPER FUNCTIONS
# ==========================================

_check_services:
	@echo "$(INFO) Service Status:"
	@$(DOCKER_COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "$(ERROR) Docker Compose not available"

_check_api_health:
	@echo "$(INFO) API Health:"
	@curl -s $(API_URL)/health >/dev/null 2>&1 && \
		echo "  $(SUCCESS) API is healthy at $(GREEN)$(API_URL)$(NC)" || \
		echo "  $(ERROR) API is not responding at $(API_URL)"

_check_database_health:
	@echo "$(INFO) Database Health:"
	@$(DOCKER_COMPOSE) exec -T db pg_isready -U postgres >/dev/null 2>&1 && \
		echo "  $(SUCCESS) Database is accepting connections" || \
		echo "  $(ERROR) Database is not ready"

_check_worker_health:
	@echo "$(INFO) Worker Health:"
	@$(DOCKER_COMPOSE) exec worker celery -A src.app.celery.worker inspect ping >/dev/null 2>&1 && \
		echo "  $(SUCCESS) Worker is responding" || \
		echo "  $(ERROR) Worker is not responding"

_check_storage_health:
	@echo "$(INFO) Storage Health:"
	@test -d "/tmp" && \
		echo "  $(SUCCESS) Local storage is available" || \
		echo "  $(ERROR) Storage directory not accessible"

_check_uploads:
	@echo "$(INFO) Upload Statistics:"
	@curl -s $(API_URL)/files/uploads 2>/dev/null | grep -o '"count":[0-9]*' | cut -d: -f2 | \
		xargs -I {} echo "  $(SUCCESS) {} uploads in database" || \
		echo "  $(WARNING) Could not retrieve upload statistics"

_wait_for_services:
	@echo "$(INFO) Waiting for services to be ready..."
	@for i in {1..30}; do \
		if curl -s $(API_URL)/health >/dev/null 2>&1; then \
			echo "$(SUCCESS) Services are ready"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "$(WARNING) Services may still be starting up"; \
		fi; \
		sleep 2; \
	done

_wait_for_infrastructure:
	@echo "$(INFO) Waiting for infrastructure to be ready..."
	@for i in {1..30}; do \
		if $(DOCKER_COMPOSE) exec -T db pg_isready -U postgres >/dev/null 2>&1; then \
			echo "$(SUCCESS) Database is ready"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "$(WARNING) Database may still be starting up"; \
		fi; \
		sleep 2; \
	done

.PHONY: help status setup dev build start stop restart logs logs-api logs-worker logs-db \
        db-shell migrate migrate-create migrate-history migrate-current migrate-rollback db-reset \
        test-upload worker-status worker-restart worker-shell file-stats \
        health debug monitor clean clean-all backup-db restore-db mcp-inspect \
        env-setup start-services dev-full dev-local \
        _check_services _check_api_health _check_database_health _check_worker_health _check_storage_health _check_uploads _wait_for_services _wait_for_infrastructure
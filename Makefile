# Makefile - PyManager Docker Management
# Optimisé pour économiser l'espace disque

.PHONY: help build up down restart logs clean prune backup restore test shell

# Variables
COMPOSE_FILE := docker-compose.yml
PROJECT_NAME := pymanager
BACKUP_DIR := ./backups
ENV_FILE := .env

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ============================================================================
# Help
# ============================================================================

help: ## Affiche cette aide
	@echo "$(BLUE)PyManager - Docker Management$(NC)"
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make [target]"
	@echo ""
	@echo "$(GREEN)Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Development
# ============================================================================

build: ## Build les images Docker (sans cache pour économiser l'espace)
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build --no-cache --pull
	@echo "$(GREEN)✓ Build complete$(NC)"

up: ## Démarre les services en arrière-plan
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo ""
	@make status

down: ## Arrête et supprime les containers
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart: down up ## Redémarre tous les services

stop: ## Arrête les services sans les supprimer
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) stop
	@echo "$(GREEN)✓ Services stopped$(NC)"

start: ## Redémarre les services arrêtés
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) start
	@echo "$(GREEN)✓ Services started$(NC)"

# ============================================================================
# Monitoring
# ============================================================================

logs: ## Affiche les logs en temps réel
	docker-compose -f $(COMPOSE_FILE) logs -f --tail=100

logs-app: ## Logs Streamlit uniquement
	docker-compose -f $(COMPOSE_FILE) logs -f streamlit-app

logs-mcp: ## Logs MCP Server uniquement
	docker-compose -f $(COMPOSE_FILE) logs -f mcp-server

logs-db: ## Logs MongoDB uniquement
	docker-compose -f $(COMPOSE_FILE) logs -f mongodb

status: ## Affiche le statut des services
	@echo "$(BLUE)Service Status:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) ps

stats: ## Affiche les statistiques (CPU, RAM)
	docker stats --no-stream $(PROJECT_NAME)-app $(PROJECT_NAME)-mcp $(PROJECT_NAME)-mongo

# ============================================================================
# Shell Access
# ============================================================================

shell: ## Ouvre un shell dans le container Streamlit
	docker exec -it $(PROJECT_NAME)-app /bin/bash

shell-mcp: ## Ouvre un shell dans le container MCP
	docker exec -it $(PROJECT_NAME)-mcp /bin/bash

shell-db: ## Ouvre mongosh dans le container MongoDB
	docker exec -it $(PROJECT_NAME)-mongo mongosh -u admin -p

# ============================================================================
# Database
# ============================================================================

backup: ## Crée un backup de la base MongoDB
	@echo "$(BLUE)Creating MongoDB backup...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@BACKUP_NAME="backup_$(shell date +%Y%m%d_%H%M%S)" && \
	docker exec $(PROJECT_NAME)-mongo mongodump \
		--username=admin \
		--password=$${MONGO_PASSWORD:-changeme} \
		--authenticationDatabase=admin \
		--out=/backups/$$BACKUP_NAME && \
	echo "$(GREEN)✓ Backup created: $(BACKUP_DIR)/$$BACKUP_NAME$(NC)"

restore: ## Restaure le dernier backup MongoDB
	@echo "$(YELLOW)Restoring MongoDB backup...$(NC)"
	@LATEST_BACKUP=$$(ls -t $(BACKUP_DIR) | head -1) && \
	if [ -z "$$LATEST_BACKUP" ]; then \
		echo "$(RED)✗ No backup found$(NC)"; \
		exit 1; \
	fi && \
	echo "Restoring: $$LATEST_BACKUP" && \
	docker exec $(PROJECT_NAME)-mongo mongorestore \
		--username=admin \
		--password=$${MONGO_PASSWORD:-changeme} \
		--authenticationDatabase=admin \
		/backups/$$LATEST_BACKUP && \
	echo "$(GREEN)✓ Restore complete$(NC)"

db-shell: shell-db ## Alias pour shell-db

# ============================================================================
# Testing
# ============================================================================

test: ## Lance les tests dans un container
	@echo "$(BLUE)Running tests...$(NC)"
	docker-compose -f $(COMPOSE_FILE) run --rm streamlit-app pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## Lance les tests avec couverture
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	docker-compose -f $(COMPOSE_FILE) run --rm streamlit-app \
		pytest tests/ -v --cov=. --cov-report=html
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

# ============================================================================
# Cleanup (Optimisé pour économiser l'espace)
# ============================================================================

clean: ## Nettoie les containers et volumes
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down -v
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-images: ## Supprime les images du projet
	@echo "$(YELLOW)Removing project images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --rmi all
	@echo "$(GREEN)✓ Images removed$(NC)"

prune: ## Nettoyage complet Docker (DANGEREUX - libère beaucoup d'espace)
	@echo "$(RED)⚠ This will remove ALL unused Docker data!$(NC)"
	@echo "Continue? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo "$(YELLOW)Pruning Docker system...$(NC)"
	docker system prune -a --volumes -f
	@echo "$(GREEN)✓ Docker system pruned$(NC)"
	@docker system df

prune-volumes: ## Supprime uniquement les volumes non utilisés
	@echo "$(YELLOW)Pruning unused volumes...$(NC)"
	docker volume prune -f
	@echo "$(GREEN)✓ Volumes pruned$(NC)"

prune-images: ## Supprime uniquement les images non utilisées
	@echo "$(YELLOW)Pruning unused images...$(NC)"
	docker image prune -a -f
	@echo "$(GREEN)✓ Images pruned$(NC)"

prune-containers: ## Supprime les containers arrêtés
	@echo "$(YELLOW)Pruning stopped containers...$(NC)"
	docker container prune -f
	@echo "$(GREEN)✓ Containers pruned$(NC)"

# ============================================================================
# Disk Management
# ============================================================================

disk-usage: ## Affiche l'utilisation disque Docker
	@echo "$(BLUE)Docker Disk Usage:$(NC)"
	@docker system df -v

size: ## Affiche la taille des images
	@echo "$(BLUE)Image Sizes:$(NC)"
	@docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | \
		grep -E "($(PROJECT_NAME)|REPOSITORY)"

volumes: ## Liste les volumes et leur taille
	@echo "$(BLUE)Docker Volumes:$(NC)"
	@docker volume ls

inspect-volume: ## Inspecte un volume spécifique
	@echo "Volume name: " && read volume && \
	docker volume inspect $$volume

# ============================================================================
# Production
# ============================================================================

deploy: build up ## Build et déploie en production
	@echo "$(GREEN)✓ Deployed!$(NC)"
	@make status

deploy-prod: ## Déploiement production avec backup
	@echo "$(BLUE)Production deployment...$(NC)"
	@make backup
	@make build
	@make down
	@make up
	@echo "$(GREEN)✓ Production deployment complete$(NC)"

# ============================================================================
# Maintenance
# ============================================================================

update: ## Met à jour les images
	@echo "$(BLUE)Updating images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) pull
	@make restart
	@echo "$(GREEN)✓ Update complete$(NC)"

rebuild: clean-images build up ## Rebuild complet des images

health: ## Vérifie la santé des services
	@echo "$(BLUE)Health Check:$(NC)"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "$(BLUE)MongoDB:$(NC)"
	@docker exec $(PROJECT_NAME)-mongo mongosh --quiet --eval "db.adminCommand('ping')" || \
		echo "$(RED)✗ MongoDB unhealthy$(NC)"
	@echo "$(BLUE)MCP Server:$(NC)"
	@curl -s http://localhost:8000/ > /dev/null && \
		echo "$(GREEN)✓ MCP Server healthy$(NC)" || \
		echo "$(RED)✗ MCP Server unhealthy$(NC)"
	@echo "$(BLUE)Streamlit:$(NC)"
	@curl -s http://localhost:8501/_stcore/health > /dev/null && \
		echo "$(GREEN)✓ Streamlit healthy$(NC)" || \
		echo "$(RED)✗ Streamlit unhealthy$(NC)"

# ============================================================================
# Development Helpers
# ============================================================================

dev: ## Mode développement (rebuild à chaque changement)
	docker-compose -f $(COMPOSE_FILE) up --build

dev-logs: ## Mode dev avec logs
	docker-compose -f $(COMPOSE_FILE) up --build

watch: ## Watch mode pour logs
	watch -n 2 make status

# ============================================================================
# Quick Actions
# ============================================================================

quick-start: up logs ## Démarrage rapide avec logs

quick-stop: down ## Arrêt rapide

quick-restart: restart logs ## Redémarrage rapide avec logs

quick-clean: down prune-containers ## Nettoyage rapide

# ============================================================================
# Info
# ============================================================================

info: ## Affiche les informations du projet
	@echo "$(BLUE)PyManager Docker Setup$(NC)"
	@echo ""
	@echo "$(GREEN)Project:$(NC) $(PROJECT_NAME)"
	@echo "$(GREEN)Compose File:$(NC) $(COMPOSE_FILE)"
	@echo "$(GREEN)Backup Directory:$(NC) $(BACKUP_DIR)"
	@echo ""
	@echo "$(GREEN)Services:$(NC)"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - MCP Server: http://localhost:8000"
	@echo "  - MongoDB: localhost:27017"
	@echo ""
	@make disk-usage

env-check: ## Vérifie les variables d'environnement
	@echo "$(BLUE)Environment Variables:$(NC)"
	@if [ -f $(ENV_FILE) ]; then \
		echo "$(GREEN)✓ .env file exists$(NC)"; \
		cat $(ENV_FILE) | grep -v "PASSWORD\|KEY\|SECRET" || true; \
	else \
		echo "$(RED)✗ .env file not found$(NC)"; \
	fi

# ============================================================================
# Default
# ============================================================================

.DEFAULT_GOAL := help

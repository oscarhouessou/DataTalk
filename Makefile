# Makefile pour DataTalk Analytics
# Usage: make start, make stop, make install, etc.

.PHONY: help install start stop clean dev api webapp

# Configuration
API_PORT = 8000
WEBAPP_PORT = 3000
VENV_PATH = venv
WEBAPP_PATH = webapp

help: ## Affiche l'aide
	@echo "🚀 DataTalk Analytics - Commandes disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

install: ## Installe toutes les dépendances
	@echo "📦 Installation des dépendances Python..."
	@if [ ! -d $(VENV_PATH) ]; then python3 -m venv $(VENV_PATH); fi
	@. $(VENV_PATH)/bin/activate && pip install -r requirements.txt
	@echo "📦 Installation des dépendances Node.js..."
	@cd $(WEBAPP_PATH) && npm install
	@echo "✅ Installation terminée!"

start: ## Lance API + WebApp ensemble
	@echo "🚀 Démarrage complet de DataTalk..."
	@./start-datatalk.sh

api: ## Lance uniquement l'API FastAPI
	@echo "🐍 Démarrage de l'API FastAPI..."
	@. $(VENV_PATH)/bin/activate && python api_simple.py

webapp: ## Lance uniquement l'application web
	@echo "⚛️  Démarrage de l'application Next.js..."
	@cd $(WEBAPP_PATH) && export PATH="/opt/homebrew/bin:$$PATH" && npm run dev

dev: start ## Alias pour 'make start'

stop: ## Arrête tous les services
	@echo "🛑 Arrêt des services..."
	@pkill -f "python api_simple.py" || true
	@pkill -f "next dev" || true
	@echo "✅ Services arrêtés"

clean: ## Nettoie les fichiers temporaires
	@echo "🧹 Nettoyage..."
	@cd $(WEBAPP_PATH) && rm -rf .next node_modules/.cache
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Nettoyage terminé"

check: ## Vérifie que les services fonctionnent
	@echo "🔍 Vérification des services..."
	@curl -s http://localhost:$(API_PORT)/sessions > /dev/null && echo "✅ API OK" || echo "❌ API non accessible"
	@curl -s http://localhost:$(WEBAPP_PORT) > /dev/null && echo "✅ WebApp OK" || echo "❌ WebApp non accessible"

logs-api: ## Affiche les logs de l'API
	@echo "📋 Logs API (Ctrl+C pour quitter):"
	@tail -f /tmp/datatalk-api.log 2>/dev/null || echo "Pas de logs API disponibles"

build: ## Build l'application pour la production
	@echo "🏗️  Build de production..."
	@cd $(WEBAPP_PATH) && npm run build
	@echo "✅ Build terminé!"

test: ## Lance les tests
	@echo "🧪 Lancement des tests..."
	@. $(VENV_PATH)/bin/activate && python -m pytest tests/ || echo "Pas de tests configurés"
	@cd $(WEBAPP_PATH) && npm run lint

status: ## Affiche le statut des services
	@echo "📊 Statut des services DataTalk:"
	@echo "API FastAPI ($(API_PORT)):"
	@lsof -ti:$(API_PORT) > /dev/null && echo "  🟢 En cours d'exécution" || echo "  🔴 Arrêté"
	@echo "WebApp Next.js ($(WEBAPP_PORT)):"
	@lsof -ti:$(WEBAPP_PORT) > /dev/null && echo "  🟢 En cours d'exécution" || echo "  🔴 Arrêté"
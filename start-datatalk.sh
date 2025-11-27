#!/bin/bash

# Script de lancement DataTalk - Lance API + WebApp automatiquement
echo "🚀 Démarrage de DataTalk Analytics..."

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction de nettoyage en cas d'arrêt
cleanup() {
    echo -e "\n${YELLOW}🛑 Arrêt de DataTalk...${NC}"
    # Tuer tous les processus en arrière-plan
    jobs -p | xargs -r kill
    echo -e "${GREEN}✅ DataTalk arrêté proprement${NC}"
    exit 0
}

# Capturer Ctrl+C pour nettoyage
trap cleanup SIGINT SIGTERM

# Vérifier si on est dans le bon répertoire
if [ ! -f "nlq.py" ]; then
    echo -e "${RED}❌ Erreur: Veuillez exécuter ce script depuis le répertoire DataTalk${NC}"
    exit 1
fi

# Vérifier l'environnement virtuel Python
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Erreur: Environnement virtuel 'venv' non trouvé${NC}"
    exit 1
fi

# Vérifier le répertoire webapp
if [ ! -d "webapp" ]; then
    echo -e "${RED}❌ Erreur: Répertoire 'webapp' non trouvé${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Vérification des dépendances...${NC}"

# Configurer le PATH pour Homebrew
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Vérifier Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js non installé${NC}"
    echo -e "${YELLOW}💡 Essayez: brew install node${NC}"
    exit 1
fi

# Vérifier npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm non installé${NC}"
    echo -e "${YELLOW}💡 Essayez: brew install node${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dépendances OK${NC}"

# 1. Lancer l'API FastAPI
echo -e "${BLUE}🐍 Démarrage de l'API FastAPI...${NC}"
source venv/bin/activate && python api_simple.py &
API_PID=$!

# Attendre que l'API soit prête
echo -e "${YELLOW}⏳ Attente de l'API (5 secondes)...${NC}"
sleep 5

# Vérifier que l'API répond
if curl -s http://localhost:8000/sessions > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API FastAPI démarrée sur http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Erreur: L'API ne répond pas${NC}"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 2. Lancer l'application Next.js
echo -e "${BLUE}⚛️  Démarrage de l'application Next.js...${NC}"
cd webapp

# Installer les dépendances si nécessaire
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installation des dépendances npm...${NC}"
    npm install
fi

# Lancer Next.js
npm run dev &
WEBAPP_PID=$!

# Attendre que l'app soit prête
echo -e "${YELLOW}⏳ Attente de l'application web (10 secondes)...${NC}"
sleep 10

echo -e "${GREEN}🎉 DataTalk Analytics est prêt !${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🌐 Application Web : http://localhost:3000${NC}"
echo -e "${GREEN}🔗 API Backend    : http://localhost:8000${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}💡 Appuyez sur Ctrl+C pour arrêter les deux services${NC}"

# Attendre indéfiniment (les processus tournent en arrière-plan)
while true; do
    sleep 1
done
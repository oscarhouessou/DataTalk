#!/bin/bash

# Script de lancement simple DataTalk
echo "🚀 Lancement rapide DataTalk..."

# Configurer le PATH
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "nlq.py" ]; then
    echo "❌ Veuillez exécuter depuis le répertoire DataTalk"
    exit 1
fi

# Fonction de nettoyage
cleanup() {
    echo "🛑 Arrêt..."
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "🐍 Lancement API..."
source venv/bin/activate && python api_simple.py &

echo "⏳ Attente API (3 sec)..."
sleep 3

echo "⚛️ Lancement WebApp..."
cd webapp && npm run dev &

echo "🎉 DataTalk prêt !"
echo "🌐 Web: http://localhost:3000"
echo "🔗 API: http://localhost:8000"
echo "💡 Ctrl+C pour arrêter"

# Attendre
while true; do sleep 1; done
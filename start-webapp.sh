#!/bin/bash

# Script de lancement de l'application web DataTalk

echo "🚀 Démarrage de DataTalk Web Application..."
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Activer le venv s'il existe
if [ -d "venv" ]; then
    echo "🔧 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

# Vérifier si les dépendances sont installées
echo "📦 Vérification des dépendances..."
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Certaines dépendances sont manquantes."
    echo "📥 Installation des dépendances..."
    pip install fastapi uvicorn python-dotenv pandas langchain langchain-openai langchain-experimental matplotlib seaborn openpyxl python-multipart
fi

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo "⚠️  Fichier .env non trouvé."
    echo "📝 Création du fichier .env..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "⚠️  Veuillez configurer votre clé API OpenAI dans le fichier .env"
    exit 1
fi

# Vérifier si la clé API est configurée
if grep -q "your_openai_api_key_here" .env; then
    echo "⚠️  Clé API OpenAI non configurée dans le fichier .env"
    echo "📝 Veuillez remplacer 'your_openai_api_key_here' par votre vraie clé API"
    exit 1
fi

echo "✅ Configuration OK"
echo ""

# Démarrer le serveur FastAPI
echo "🌐 Démarrage du serveur API sur http://localhost:8000"
echo "📱 Interface web disponible sur http://localhost:8000/web/index.html"
echo ""
echo "💡 Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

# Ouvrir le navigateur après 2 secondes
(sleep 2 && open http://localhost:8000/web/index.html) &

# Lancer le serveur
python api_simple.py


@echo off
:: Script de lancement DataTalk pour Windows
echo 🚀 Démarrage de DataTalk Analytics...

:: Vérifier si on est dans le bon répertoire
if not exist "nlq.py" (
    echo ❌ Erreur: Veuillez exécuter ce script depuis le répertoire DataTalk
    pause
    exit /b 1
)

:: Vérifier l'environnement virtuel
if not exist "venv" (
    echo ❌ Erreur: Environnement virtuel 'venv' non trouvé
    pause
    exit /b 1
)

:: Vérifier le répertoire webapp
if not exist "webapp" (
    echo ❌ Erreur: Répertoire 'webapp' non trouvé
    pause
    exit /b 1
)

echo 🔧 Vérification des dépendances...

:: Lancer l'API FastAPI en arrière-plan
echo 🐍 Démarrage de l'API FastAPI...
start "DataTalk API" /min cmd /c "venv\Scripts\activate && python api_simple.py"

:: Attendre que l'API soit prête
echo ⏳ Attente de l'API (8 secondes)...
timeout /t 8 /nobreak > nul

:: Lancer l'application Next.js
echo ⚛️  Démarrage de l'application Next.js...
cd webapp

:: Installer les dépendances si nécessaire
if not exist "node_modules" (
    echo 📦 Installation des dépendances npm...
    npm install
)

:: Lancer Next.js
echo 🎉 DataTalk Analytics est prêt !
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🌐 Application Web : http://localhost:3000
echo 🔗 API Backend    : http://localhost:8000
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 💡 Fermez cette fenêtre pour arrêter les services

npm run dev

pause
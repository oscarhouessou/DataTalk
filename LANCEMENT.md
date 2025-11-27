# 🚀 DataTalk Analytics - Lancement Automatique

## Démarrage Rapide

### Option 1: Script automatique (Recommandé)
```bash
# macOS/Linux
./start-datatalk.sh

# Windows
start-datatalk.bat
```

### Option 2: Makefile (Développeurs)
```bash
make start          # Lance API + WebApp
make install        # Installe les dépendances
make stop          # Arrête tout
make status        # Vérifie l'état
make help          # Aide complète
```

### Option 3: Commandes manuelles
```bash
# Terminal 1 - API
cd DataTalk
source venv/bin/activate
python api_simple.py

# Terminal 2 - WebApp
cd DataTalk/webapp
export PATH="/opt/homebrew/bin:$PATH"
npm run dev
```

## 📍 URLs d'accès

- **🌐 Application Web**: http://localhost:3000
- **🔗 API Backend**: http://localhost:8000
- **📚 Documentation API**: http://localhost:8000/docs

## 🔧 Configuration

### Variables d'environnement (webapp/.env.local)
```bash
API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="DataTalk Analytics"
```

### Ports utilisés
- **Frontend Next.js**: 3000
- **Backend FastAPI**: 8000

## 🛠️ Développement

### Structure du projet
```
DataTalk/
├── start-datatalk.sh      # 🚀 Script de lancement Unix
├── start-datatalk.bat     # 🚀 Script de lancement Windows  
├── Makefile              # 🛠️ Commandes développeur
├── api_simple.py         # 🐍 API FastAPI
├── nlq.py               # 📊 Application Streamlit originale
├── requirements.txt     # 📦 Dépendances Python
└── webapp/              # ⚛️ Application Next.js
    ├── package.json     # 📦 Dépendances Node.js
    ├── src/
    │   ├── app/         # 🏠 Pages Next.js
    │   ├── components/  # 🧩 Composants React
    │   └── hooks/       # 🎣 Hooks personnalisés
    └── .env.local       # ⚙️ Configuration
```

## 🚨 Dépannage

### Problèmes courants

1. **Port déjà utilisé**:
   ```bash
   make stop  # ou tuer les processus manuellement
   lsof -ti:3000 | xargs kill  # WebApp
   lsof -ti:8000 | xargs kill  # API
   ```

2. **Dépendances manquantes**:
   ```bash
   make install  # Réinstalle tout
   ```

3. **API non accessible**:
   ```bash
   make check    # Vérifie l'état des services
   make logs-api # Affiche les logs
   ```

## ⚡ Commandes utiles

```bash
# Démarrage complet
make start

# Vérification
make check
make status

# Nettoyage
make clean
make stop

# Build production
make build

# Tests
make test
```
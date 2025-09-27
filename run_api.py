#!/usr/bin/env python3
"""
Script de démarrage pour l'API DataTalk
Usage: python run_api.py
"""

import os
import sys
import uvicorn
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration par défaut
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_RELOAD = True

def main():
    """Point d'entrée principal"""
    
    # Configuration depuis les variables d'environnement
    host = os.getenv("API_HOST", DEFAULT_HOST)
    port = int(os.getenv("API_PORT", DEFAULT_PORT))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))
    log_level = os.getenv("API_LOG_LEVEL", "info").lower()
    
    print(f"""
🚀 Démarrage de DataTalk API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📡 URL: http://{host}:{port}
📚 Documentation: http://{host}:{port}/docs
🔄 Rechargement auto: {'✅ Activé' if reload else '❌ Désactivé'}
👥 Workers: {workers}
📝 Niveau de log: {log_level.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Endpoints disponibles:
   • POST /api/v1/data/upload - Upload de données
   • POST /api/v1/chat/query - Requêtes NLQ
   • POST /api/v1/questions/suggest - Questions suggérées
   • POST /api/v1/insights/generate - Insights automatiques
   • POST /api/v1/charts/create - Création de graphiques
   • GET  /api/v1/sessions - Gestion des sessions
   
🔒 Authentification requise via header: Authorization: Bearer YOUR_API_KEY
""")
    
    # Configuration d'uvicorn
    uvicorn_config = {
        "app": "api.main:app",
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": log_level,
        "access_log": True,
    }
    
    # Ajouter workers seulement en production
    if not reload and workers > 1:
        uvicorn_config["workers"] = workers
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'API DataTalk")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
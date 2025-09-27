# 🚀 DataTalk API - Documentation Complète

## Vue d'ensemble

L'API DataTalk est une interface REST permettant de traiter des données en langage naturel avec des capacités d'intelligence artificielle avancées. Elle réutilise toute la logique métier du code `nlq.py` existant sans le modifier, offrant une architecture microservices évolutive.

## 🏗️ Architecture

```
DataTalk/
├── nlq.py                  # Application Streamlit existante (NON MODIFIÉE)
├── api/                    # Nouvelle API FastAPI
│   ├── main.py            # Point d'entrée FastAPI
│   ├── models/            # Modèles Pydantic
│   ├── services/          # Logique métier réutilisée
│   │   ├── nlq_service.py     # Traitement NLQ
│   │   ├── chart_service.py   # Génération graphiques
│   │   └── insights_service.py # Insights automatiques
│   └── config/            # Configuration
├── run_api.py             # Script de démarrage
└── test_api.py           # Tests automatisés
```

## 🚀 Installation et Démarrage

### Prérequis
- Python 3.8+
- Dépendances du projet nlq.py déjà installées
- Clé API OpenAI configurée

### Installation des dépendances API
```bash
pip install -r api/requirements.txt
```

### Configuration
Ajouter à votre fichier `.env` :
```bash
# API Configuration
API_KEY=your-secure-api-key-here
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Déjà configuré pour nlq.py
OPENAI_API_KEY=your-openai-key
```

### Démarrage de l'API
```bash
# Méthode 1: Script de démarrage
python run_api.py

# Méthode 2: Direct uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera disponible sur :
- 🌐 Interface : http://localhost:8000
- 📚 Documentation : http://localhost:8000/docs
- 🔧 Redoc : http://localhost:8000/redoc

## 🔒 Authentification

Toutes les requêtes (sauf `/health`) nécessitent un token d'authentification :

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/api/v1/...
```

## 📋 Endpoints Disponibles

### 1. 📊 Upload de Données
**POST** `/api/v1/data/upload`

Upload d'un fichier de données (CSV, Excel, JSON).

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@data.csv" \
  -F "session_id=my-session-123" \
  http://localhost:8000/api/v1/data/upload
```

**Réponse :**
```json
{
  "success": true,
  "session_id": "my-session-123",
  "filename": "data.csv",
  "rows": 1000,
  "columns": 5,
  "column_names": ["nom", "age", "salaire", "departement"],
  "data_types": {"nom": "object", "age": "int64", ...},
  "preview": [...],
  "automatic_insights": [...]
}
```

### 2. 💬 Requête en Langage Naturel
**POST** `/api/v1/chat/query`

Traite une question en langage naturel sur vos données.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "query": "Quel est le salaire moyen par département?",
    "context": "Analyse RH"
  }' \
  http://localhost:8000/api/v1/chat/query
```

**Réponse :**
```json
{
  "success": true,
  "session_id": "my-session-123",
  "query": "Quel est le salaire moyen par département?",
  "answer": "Le salaire moyen par département est...",
  "code_executed": "df.groupby('departement')['salaire'].mean()",
  "has_chart": false,
  "execution_time": 1.23
}
```

### 3. ❓ Suggestions de Questions
**POST** `/api/v1/questions/suggest`

Génère des questions intelligentes basées sur vos données.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "context": "Dataset RH",
    "num_questions": 5
  }' \
  http://localhost:8000/api/v1/questions/suggest
```

**Réponse :**
```json
{
  "success": true,
  "session_id": "my-session-123",
  "questions": [
    {
      "question": "Quelle est la distribution des salaires?",
      "category": "exploration"
    },
    {
      "question": "Y a-t-il une corrélation entre âge et salaire?",
      "category": "analyse"
    }
  ]
}
```

### 4. 🔍 Génération d'Insights
**POST** `/api/v1/insights/generate`

Détecte automatiquement des insights dans vos données.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "analysis_type": "comprehensive"
  }' \
  http://localhost:8000/api/v1/insights/generate
```

**Réponse :**
```json
{
  "success": true,
  "session_id": "my-session-123",
  "insights": [
    {
      "type": "anomalie",
      "title": "Salaires déséquilibrés",
      "description": "Écart important entre départements...",
      "severity": "high",
      "recommendation": "Réviser la politique salariale"
    }
  ],
  "analysis_type": "comprehensive"
}
```

### 5. 📈 Création de Graphiques
**POST** `/api/v1/charts/create`

Crée des visualisations intelligentes.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "query": "Graphique des salaires par département",
    "chart_type": "bar",
    "columns": ["departement", "salaire"]
  }' \
  http://localhost:8000/api/v1/charts/create
```

**Réponse :**
```json
{
  "success": true,
  "session_id": "my-session-123",
  "chart_type": "bar",
  "chart_data": {
    "image": "base64_encoded_image...",
    "format": "png",
    "width": 1200,
    "height": 700
  },
  "description": "Salaires par département",
  "columns_used": ["departement", "salaire"]
}
```

### 6. 📋 Gestion des Sessions
**GET** `/api/v1/sessions`

Liste toutes les sessions actives.

**GET** `/api/v1/sessions/{session_id}/history`

Récupère l'historique d'une session.

**DELETE** `/api/v1/sessions/{session_id}`

Supprime une session et ses données.

## 🧪 Tests Automatisés

Testez l'API avec le script intégré :

```bash
python test_api.py
```

Ce script teste tous les endpoints et affiche un rapport détaillé.

## 🛠️ Intégration avec Frontend

### JavaScript/React Example
```javascript
class DataTalkAPI {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async uploadData(file, sessionId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await fetch(`${this.baseUrl}/api/v1/data/upload`, {
      method: 'POST',
      headers: { 'Authorization': this.headers.Authorization },
      body: formData
    });

    return response.json();
  }

  async query(sessionId, question) {
    const response = await fetch(`${this.baseUrl}/api/v1/chat/query`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        session_id: sessionId,
        query: question
      })
    });

    return response.json();
  }
}

// Usage
const api = new DataTalkAPI('http://localhost:8000', 'your-api-key');
const result = await api.query('session-123', 'Montrez-moi les ventes par région');
```

### Python Client Example
```python
import requests

class DataTalkClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def upload_data(self, file_path, session_id):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'session_id': session_id}
            response = requests.post(
                f"{self.base_url}/api/v1/data/upload",
                files=files,
                data=data,
                headers=self.headers
            )
        return response.json()
    
    def query(self, session_id, question):
        payload = {
            "session_id": session_id,
            "query": question
        }
        response = requests.post(
            f"{self.base_url}/api/v1/chat/query",
            json=payload,
            headers=self.headers
        )
        return response.json()

# Usage
client = DataTalkClient("http://localhost:8000", "your-api-key")
result = client.query("session-123", "Analysez les tendances de vente")
```

## 🔧 Configuration Avancée

### Variables d'Environnement Complètes
```bash
# API Core
API_KEY=your-secure-api-key
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
API_WORKERS=1

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# Limites
MAX_FILE_SIZE=104857600  # 100MB
MAX_ROWS=1000000
SESSION_TIMEOUT=3600

# Base de données (futur)
DATABASE_URL=postgresql://user:pass@localhost/datatalk
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Déploiement Production

#### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r api/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "run_api.py"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  datatalk-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_KEY=${API_KEY}
    volumes:
      - ./data:/app/data
```

## 🚀 Fonctionnalités Avancées

### 1. Cache et Performance
- Cache Redis pour les résultats fréquents
- Limitation du taux de requêtes
- Compression des réponses

### 2. Monitoring
- Logs structurés avec timestamps
- Métriques de performance
- Alertes en cas d'erreur

### 3. Sécurité
- Authentification par token
- Validation stricte des inputs
- Limitation de taille des fichiers
- CORS configurables

### 4. Évolutivité
- Architecture microservices
- Sessions persistantes
- Support multi-utilisateurs
- API versioning

## 🤝 Intégration avec Streamlit Existant

L'API **ne modifie en rien** le code `nlq.py` existant. Les deux peuvent fonctionner simultanément :

- **Streamlit** (port 8501) : Interface utilisateur interactive
- **API FastAPI** (port 8000) : Interface programmatique

### Utilisation Conjointe
```python
# Streamlit continue de fonctionner normalement
streamlit run nlq.py

# API disponible pour intégrations externes  
python run_api.py

# Les deux partagent la même logique métier !
```

## 🎯 Cas d'Usage Business

### 1. Intégration dans Applications Existantes
- Dashboards personnalisés
- Applications mobiles
- Outils internes d'entreprise

### 2. Automatisation
- Rapports automatiques
- Alertes basées sur les données
- Pipelines d'analyse

### 3. API B2B
- SaaS multi-tenant
- Marketplace de données
- Services d'analyse externalisés

## 💡 Roadmap

### Version 1.1
- [ ] Base de données PostgreSQL
- [ ] Cache Redis intégré
- [ ] Authentification multi-utilisateurs

### Version 1.2
- [ ] WebSocket pour temps réel
- [ ] Export de rapports PDF
- [ ] Intégration CI/CD

### Version 2.0
- [ ] Interface admin web
- [ ] Analytics et métriques
- [ ] Marketplace de templates

## 🆘 Support et Dépannage

### Problèmes Courants

**L'API ne démarre pas :**
```bash
# Vérifier les dépendances
pip install -r api/requirements.txt

# Vérifier le port
netstat -an | grep 8000

# Logs détaillés
API_LOG_LEVEL=DEBUG python run_api.py
```

**Erreurs d'authentification :**
```bash
# Vérifier la clé API dans .env
echo $API_KEY

# Test sans authentification
curl http://localhost:8000/health
```

**Erreurs OpenAI :**
```bash
# Vérifier la clé
echo $OPENAI_API_KEY

# Tester directement
python -c "import openai; print('OK')"
```

### Logs et Monitoring
```bash
# Voir les logs en temps réel
tail -f datatalk_api.log

# Métriques de performance
curl http://localhost:8000/health
```

---

## 🎉 Conclusion

L'API DataTalk offre une interface professionnelle et évolutive pour vos analyses de données en langage naturel, tout en préservant parfaitement votre application Streamlit existante. 

**Architecture gagnante :**
- ✅ Code nlq.py **non modifié**
- ✅ Logique métier **réutilisée** 
- ✅ API **professionnelle** pour intégrations
- ✅ Streamlit **conservé** pour usage interactif
- ✅ **Évolutivité** assurée pour le business

Prêt pour la **commercialisation** et l'**intégration** dans vos écosystèmes clients ! 🚀
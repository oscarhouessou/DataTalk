# 🚀 Guide de Déploiement DataTalk sur Cloud Run

Ce guide vous explique comment déployer automatiquement votre application DataTalk sur Google Cloud Run via GitHub Actions.

## 📋 Prérequis

1. **Compte Google Cloud Platform** avec facturation activée
2. **Repository GitHub** avec votre code DataTalk
3. **Clé API OpenAI** valide

## ⚙️ Configuration Initiale

### 1. Configuration Google Cloud

```bash
# 1. Connectez-vous à gcloud
gcloud auth login

# 2. Créez un projet (ou utilisez un existant)
gcloud projects create mon-projet-datatalk
gcloud config set project mon-projet-datatalk

# 3. Exécutez le script de configuration
./setup-gcp-permissions.sh mon-projet-datatalk
```

### 2. Configuration GitHub Secrets

Dans votre repository GitHub, allez dans **Settings > Secrets and variables > Actions** et ajoutez :

| Secret | Valeur | Description |
|--------|--------|-------------|
| `GCP_PROJECT_ID` | `mon-projet-datatalk` | ID de votre projet GCP |
| `GCP_SA` | `{contenu du fichier JSON}` | Clé du service account (format base64) |
| `OPENAI_API_KEY` | `sk-proj-...` | Votre clé API OpenAI |
| `GROQ_API_KEY` | `gsk_...` | Votre clé API Groq |

## 🎯 Déploiement Automatique

### Déclenchement

Le déploiement se lance automatiquement quand vous :
- Poussez du code sur la branche `main`
- Déclenchez manuellement le workflow depuis l'onglet Actions

### Process de Déploiement

1. **Build** : Construction de l'image Docker
2. **Push** : Envoi vers Google Container Registry
3. **Deploy** : Déploiement sur Cloud Run avec les variables d'environnement

### Configuration Cloud Run

- **Port** : 8000
- **Mémoire** : 2 GiB
- **CPU** : 1 vCPU
- **Instances max** : 10
- **Timeout** : 300 secondes
- **Accès** : Public (sans authentification)

## 🌐 Accès à l'Application

Une fois déployée, votre application sera disponible sur :

```
https://datatalk-app-[hash].europe-west1.run.app
```

L'interface web sera accessible via :

```
https://datatalk-app-[hash].europe-west1.run.app/web/index.html
```

## 🔧 Variables d'Environnement

| Variable | Source | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | GitHub Secret | Clé API OpenAI pour l'IA |
| `GROQ_API_KEY` | GitHub Secret | Clé API Groq pour l'IA |
| `PORT` | Cloud Run | Port d'écoute (8000) |

## 📊 Monitoring

### Logs
```bash
# Voir les logs de l'application
gcloud run services logs read datatalk-app --region=europe-west1
```

### Métriques
- CPU et mémoire dans la console Cloud Run
- Requêtes et latence
- Erreurs 4xx/5xx

## 🛠️ Dépannage

### Erreurs Communes

**Build Failed**
```bash
# Vérifiez les logs de build
gcloud builds list --limit=5
gcloud builds log BUILD_ID
```

**Deployment Failed**
```bash
# Vérifiez les permissions
gcloud projects get-iam-policy PROJECT_ID
```

**Application ne démarre pas**
```bash
# Vérifiez les logs de runtime
gcloud run services logs read datatalk-app --region=europe-west1
```

### Variables d'Environnement Manquantes

Si `OPENAI_API_KEY` n'est pas configurée :
1. Vérifiez le secret GitHub
2. Redéployez le workflow

## 🔄 Mise à Jour

Pour mettre à jour l'application :
1. Modifiez votre code
2. Committez et poussez sur `main`
3. Le déploiement se fera automatiquement

## 💰 Coûts

Cloud Run facture à l'usage :
- **CPU** : ~0,024€ par vCPU-heure
- **Mémoire** : ~0,0025€ par GiB-heure
- **Requêtes** : Gratuit jusqu'à 2M/mois

Estimation pour usage modéré : **5-20€/mois**

## 📞 Support

- **Logs d'erreur** : Console Google Cloud > Cloud Run > datatalk-app
- **GitHub Actions** : Onglet Actions de votre repository
- **API Issues** : Vérifiez votre quota OpenAI

---

🎉 **Votre application DataTalk est maintenant déployée et accessible mondialement !**
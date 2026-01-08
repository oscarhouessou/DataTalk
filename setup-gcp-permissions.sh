#!/bin/bash

# Script de configuration des permissions pour le déploiement DataTalk sur Cloud Run

if [ $# -eq 0 ]; then
    echo "Usage: ./setup-gcp-permissions.sh PROJECT_ID [SERVICE_ACCOUNT_EMAIL]"
    echo ""
    echo "Exemple:"
    echo "  ./setup-gcp-permissions.sh mon-projet-123"
    echo "  ./setup-gcp-permissions.sh mon-projet-123 github-actions@mon-projet-123.iam.gserviceaccount.com"
    exit 1
fi

PROJECT_ID=$1
SERVICE_ACCOUNT_EMAIL=${2:-"github-actions@${PROJECT_ID}.iam.gserviceaccount.com"}

echo "🚀 Configuration des permissions pour DataTalk sur Cloud Run"
echo "📋 Projet: $PROJECT_ID"
echo "👤 Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""

# Définir le projet par défaut
gcloud config set project $PROJECT_ID

echo "1️⃣ Activation des APIs nécessaires..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

echo ""
echo "2️⃣ Création du service account (si il n'existe pas)..."
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Service Account" \
    --description="Service account pour déploiement automatique via GitHub Actions" 2>/dev/null || echo "Service account existe déjà"

echo ""
echo "3️⃣ Attribution des rôles nécessaires..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/artifactregistry.admin"

echo ""
echo "4️⃣ Génération de la clé du service account..."
KEY_FILE="datatalk-service-account-key.json"
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

echo ""
echo "✅ Configuration terminée !"
echo ""
echo "📝 Prochaines étapes dans GitHub:"
echo "1. Allez dans Settings > Secrets and variables > Actions"
echo "2. Ajoutez les secrets suivants:"
echo "   • GCP_PROJECT_ID: $PROJECT_ID"
echo "   • GCP_SA: $(cat $KEY_FILE | base64 | tr -d '\n')"
echo "   • OPENAI_API_KEY: votre_clé_openai_api"
echo "   • GROQ_API_KEY: votre_clé_groq_api"
echo ""
echo "🔑 Fichier de clé généré: $KEY_FILE"
echo "⚠️  IMPORTANT: Supprimez ce fichier après avoir configuré GitHub Secrets"
echo ""
echo "🚀 Votre application sera déployée sur:"
echo "   https://datatalk-app-[hash].europe-west1.run.app"
"""
FastAPI backend pour DataTalk - API de traitement de données en langage naturel
Architecture microservices sans modification du code Streamlit existant
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import io
import logging
from datetime import datetime

# Import des services
from .services.nlq_service import NLQService
from .services.chart_service import ChartService
from .services.insights_service import InsightsService
from .models.api_models import (
    QueryRequest, 
    QueryResponse, 
    ChartRequest, 
    ChartResponse, 
    InsightsRequest, 
    InsightsResponse,
    QuestionsRequest,
    QuestionsResponse,
    DataUploadResponse
)
from .config.settings import get_settings

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="DataTalk API",
    description="API de traitement de données en langage naturel avec IA avancée",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer selon vos besoins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité
security = HTTPBearer()
settings = get_settings()

# Services globaux
nlq_service = NLQService()
chart_service = ChartService()
insights_service = InsightsService()

# Stockage temporaire des datasets (à remplacer par une base de données)
datasets: Dict[str, pd.DataFrame] = {}
user_sessions: Dict[str, List[Dict]] = {}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérification du token d'authentification"""
    # Implémentation basique - à améliorer selon vos besoins
    if settings.api_key and credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401, 
            detail="Token d'authentification invalide"
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Bienvenue sur DataTalk API", 
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "nlq_service": "active",
            "chart_service": "active",
            "insights_service": "active"
        }
    }

@app.post("/api/v1/data/upload", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    session_id: str = "default",
    token: str = Depends(verify_token)
):
    """Upload d'un fichier de données (CSV, Excel, JSON)"""
    try:
        # Vérification du type de fichier
        if file.content_type not in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/json"]:
            raise HTTPException(status_code=400, detail="Type de fichier non supporté")
        
        # Lecture du fichier
        contents = await file.read()
        
        if file.content_type == "text/csv":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(io.BytesIO(contents))
        elif file.content_type == "application/json":
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        
        # Stockage du dataset
        datasets[session_id] = df
        
        # Génération d'insights automatiques
        auto_insights = await insights_service.detect_automatic_insights(df)
        
        logger.info(f"Dataset uploadé pour la session {session_id}: {df.shape}")
        
        return DataUploadResponse(
            session_id=session_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            preview=df.head().to_dict('records'),
            automatic_insights=auto_insights
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier: {str(e)}")

@app.post("/api/v1/chat/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Traitement d'une requête en langage naturel"""
    try:
        # Vérification de l'existence du dataset
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé pour cette session")
        
        df = datasets[request.session_id]
        
        # Traitement de la requête avec le service NLQ
        response = await nlq_service.process_query(
            query=request.query,
            dataframe=df,
            chat_history=user_sessions.get(request.session_id, [])
        )
        
        # Sauvegarde de l'historique
        if request.session_id not in user_sessions:
            user_sessions[request.session_id] = []
        
        user_sessions[request.session_id].extend([
            {"role": "user", "content": request.query, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response.answer, "timestamp": datetime.now().isoformat()}
        ])
        
        logger.info(f"Requête traitée pour la session {request.session_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/api/v1/questions/suggest", response_model=QuestionsResponse)
async def suggest_questions(
    request: QuestionsRequest,
    token: str = Depends(verify_token)
):
    """Génération de questions intelligentes sur les données"""
    try:
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé pour cette session")
        
        df = datasets[request.session_id]
        
        # Génération des questions avec le service NLQ
        questions = await nlq_service.generate_smart_questions(
            dataframe=df,
            context=request.context
        )
        
        logger.info(f"Questions générées pour la session {request.session_id}")
        
        return QuestionsResponse(
            session_id=request.session_id,
            questions=questions
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/api/v1/insights/generate", response_model=InsightsResponse)
async def generate_insights(
    request: InsightsRequest,
    token: str = Depends(verify_token)
):
    """Génération d'insights automatiques sur les données"""
    try:
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé pour cette session")
        
        df = datasets[request.session_id]
        
        # Génération des insights
        insights = await insights_service.detect_automatic_insights(
            dataframe=df,
            analysis_type=request.analysis_type
        )
        
        logger.info(f"Insights générés pour la session {request.session_id}")
        
        return InsightsResponse(
            session_id=request.session_id,
            insights=insights,
            analysis_type=request.analysis_type
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/api/v1/charts/create", response_model=ChartResponse)
async def create_chart(
    request: ChartRequest,
    token: str = Depends(verify_token)
):
    """Création de graphiques intelligents"""
    try:
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé pour cette session")
        
        df = datasets[request.session_id]
        
        # Création du graphique avec le service
        chart_response = await chart_service.create_chart_from_ai_recommendation(
            dataframe=df,
            query=request.query,
            chart_type=request.chart_type,
            columns=request.columns
        )
        
        logger.info(f"Graphique créé pour la session {request.session_id}")
        
        return chart_response
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du graphique: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/v1/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Récupération de l'historique d'une session"""
    if session_id not in user_sessions:
        return {"session_id": session_id, "history": []}
    
    return {
        "session_id": session_id,
        "history": user_sessions[session_id]
    }

@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Suppression d'une session et de ses données"""
    deleted_items = []
    
    if session_id in datasets:
        del datasets[session_id]
        deleted_items.append("dataset")
    
    if session_id in user_sessions:
        del user_sessions[session_id]
        deleted_items.append("chat_history")
    
    return {
        "message": f"Session {session_id} supprimée",
        "deleted_items": deleted_items
    }

@app.get("/api/v1/sessions")
async def list_sessions(token: str = Depends(verify_token)):
    """Liste de toutes les sessions actives"""
    sessions_info = []
    
    for session_id in datasets.keys():
        df = datasets[session_id]
        history_count = len(user_sessions.get(session_id, []))
        
        sessions_info.append({
            "session_id": session_id,
            "dataset_shape": df.shape,
            "columns": list(df.columns),
            "chat_messages": history_count,
            "last_activity": user_sessions.get(session_id, [{}])[-1].get("timestamp") if user_sessions.get(session_id) else None
        })
    
    return {"sessions": sessions_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
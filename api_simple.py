"""
API DataTalk Simplifiée - Version fonctionnelle
Réutilise la logique de nlq.py sans le modifier
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
from dotenv import load_dotenv
import sys
import importlib.util

# Charger les variables d'environnement
load_dotenv()

# Créer l'instance FastAPI
app = FastAPI(
    title="DataTalk API",
    description="API de traitement de données en langage naturel",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage temporaire des datasets
datasets = {}

# Importer les fonctions du nlq.py existant
def import_nlq_functions():
    """Importe les fonctions du fichier nlq.py sans l'exécuter"""
    spec = importlib.util.spec_from_file_location("nlq_functions", "nlq.py")
    nlq_module = importlib.util.module_from_spec(spec)
    
    # Remplacer streamlit temporairement pour éviter les erreurs
    import sys
    class MockStreamlit:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    sys.modules['streamlit'] = MockStreamlit()
    
    try:
        spec.loader.exec_module(nlq_module)
        return nlq_module
    except Exception as e:
        print(f"Erreur lors de l'import de nlq.py: {e}")
        return None

# Charger les fonctions NLQ
nlq_functions = import_nlq_functions()

# Modèles Pydantic
class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    success: bool = True

class QuestionsRequest(BaseModel):
    session_id: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list
    success: bool = True

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "DataTalk API", 
        "version": "1.0.0",
        "status": "active",
        "endpoints": ["/docs", "/upload", "/chat", "/questions"]
    }

@app.get("/health")
async def health():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Upload d'un fichier de données"""
    try:
        # Lire le fichier
        contents = await file.read()
        
        # Détecter le type de fichier principalement par extension
        filename = file.filename.lower() if file.filename else ""
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            # Fallback sur content-type si extension non reconnue
            if "csv" in str(file.content_type).lower():
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            else:
                raise HTTPException(status_code=400, detail=f"Extension non supportée: {filename}. Utilisez .csv, .xlsx, .xls ou .json")
        
        # Stocker le dataset
        datasets[session_id] = df
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """Traitement d'une requête en langage naturel"""
    try:
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé")
        
        df = datasets[request.session_id]
        
        # Utiliser l'agent NLQ du nlq.py
        from langchain_openai import ChatOpenAI
        from langchain_experimental.agents import create_pandas_dataframe_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Créer l'agent avec le même prompt que nlq.py
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, agent_type="openai-functions", 
            allow_dangerous_code=True,
            prefix="""
            Tu es un DATA ANALYST EXPERT de niveau senior avec 10+ années d'expérience.
            Tu travailles avec un DataFrame pandas pour répondre aux questions business.
            Sois factuel, précis et actionnable dans tes réponses.
            """
        )
        
        # Exécuter la requête
        result = agent.invoke(request.query)
        answer = result.get("output", "")
        
        return QueryResponse(
            session_id=request.session_id,
            query=request.query,
            answer=answer
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/questions")
async def suggest_questions(request: QuestionsRequest):
    """Génération de questions suggérées"""
    try:
        if request.session_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset non trouvé")
        
        df = datasets[request.session_id]
        
        if nlq_functions and hasattr(nlq_functions, 'generate_smart_questions'):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            
            questions = nlq_functions.generate_smart_questions(llm, df)
            
            return {
                "session_id": request.session_id,
                "questions": questions,
                "success": True
            }
        else:
            # Questions par défaut
            return {
                "session_id": request.session_id,
                "questions": [
                    "Quelles sont les statistiques principales de ce dataset?",
                    "Y a-t-il des valeurs manquantes?",
                    "Comment les données sont-elles distribuées?"
                ],
                "success": True
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """Liste des sessions actives"""
    sessions_info = []
    for session_id, df in datasets.items():
        sessions_info.append({
            "session_id": session_id,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        })
    
    return {"sessions": sessions_info}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Suppression d'une session"""
    if session_id in datasets:
        del datasets[session_id]
        return {"message": f"Session {session_id} supprimée"}
    else:
        raise HTTPException(status_code=404, detail="Session non trouvée")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
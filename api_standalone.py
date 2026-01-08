"""
API DataTalk Simplifiée - Version sans dépendance nlq.py
Version autonome qui évite les conflits de dépendances
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import os
from dotenv import load_dotenv
import base64
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour les graphiques
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

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

# Monter le dossier web pour servir les fichiers statiques
if os.path.exists("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Stockage temporaire des datasets et agents
datasets = {}
agents = {}

def get_llm():
    """Initialise le LLM Groq"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile", 
            temperature=0.1,
            openai_api_key=groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1"
        )
    else:
        print("❌ GROQ_API_KEY non configurée !")
        return None

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

def create_chart_if_needed(df, query, answer):
    """Crée un graphique si nécessaire basé sur la requête et la réponse"""
    try:
        # Configuration matplotlib
        plt.style.use('default')
        plt.figure(figsize=(10, 6))
        
        # Logique simplifiée pour déterminer le type de graphique
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        chart_created = False
        
        # Graphique en barres pour les données catégorielles
        if any(word in query.lower() for word in ['distribution', 'répartition', 'par catégorie', 'top']) and categorical_cols:
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            plt.figure(figsize=(12, 6))
            value_counts.plot(kind='bar')
            plt.title(f'Distribution de {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_created = True
            
        # Graphique linéaire pour les tendances temporelles
        elif any(word in query.lower() for word in ['évolution', 'tendance', 'temps', 'année']) and numeric_cols:
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df[numeric_cols[0]])
                plt.title(f'Évolution de {numeric_cols[0]}')
                plt.xlabel('Index')
                plt.ylabel(numeric_cols[0])
                plt.tight_layout()
                chart_created = True
                
        # Histogramme pour les distributions numériques
        elif any(word in query.lower() for word in ['histogramme', 'distribution', 'répartition']) and numeric_cols:
            plt.figure(figsize=(10, 6))
            plt.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7)
            plt.title(f'Distribution de {numeric_cols[0]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Fréquence')
            plt.tight_layout()
            chart_created = True
        
        if chart_created:
            # Sauvegarder le graphique en base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_data}"
    
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")
    
    return None

def generate_smart_questions(df):
    """Génère des suggestions de questions intelligentes"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    questions = []
    
    if categorical_cols:
        questions.append(f"Quelle est la distribution de {categorical_cols[0]} ?")
        if len(categorical_cols) > 1:
            questions.append(f"Comment se répartissent les données par {categorical_cols[1]} ?")
    
    if numeric_cols:
        questions.append(f"Quelle est la moyenne de {numeric_cols[0]} ?")
        if len(numeric_cols) > 1:
            questions.append(f"Y a-t-il une corrélation entre {numeric_cols[0]} et {numeric_cols[1]} ?")
    
    questions.extend([
        "Combien de lignes et de colonnes contient ce dataset ?",
        "Quelles sont les valeurs manquantes dans les données ?"
    ])
    
    return questions[:6]  # Limiter à 6 questions

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "DataTalk API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload et traitement d'un fichier CSV/Excel"""
    try:
        # Lire le contenu du fichier
        content = await file.read()
        
        # Déterminer le type de fichier et le charger
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")
        
        # Nettoyer les données
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Stocker le dataset
        datasets[session_id] = {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': pd.Timestamp.now()
        }
        
        # Créer l'agent pandas
        try:
            llm = get_llm()
            
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=True,
                return_intermediate_steps=True,
                allow_dangerous_code=True
            )
            
            agents[session_id] = agent
        except Exception as e:
            print(f"Erreur lors de la création de l'agent: {e}")
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            success=True
        )
        
    except Exception as e:
        print(f"Erreur upload: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Traitement d'une requête en langage naturel"""
    try:
        session_id = request.session_id
        query = request.query
        
        if session_id not in datasets:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        df = datasets[session_id]['dataframe']
        
        # Traitement avec l'agent si disponible
        if session_id in agents:
            try:
                agent = agents[session_id]
                result = agent.invoke(query)
                answer = str(result['output'])
            except Exception as e:
                print(f"Erreur agent: {e}")
                answer = f"Analyse basique: Le dataset contient {len(df)} lignes et {len(df.columns)} colonnes."
        else:
            # Réponse basique sans agent
            answer = f"Le dataset contient {len(df)} lignes et {len(df.columns)} colonnes avec les colonnes: {', '.join(df.columns.tolist())}"
        
        # Créer un graphique si approprié
        chart_data = create_chart_if_needed(df, query, answer)
        
        response_data = {
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "success": True
        }
        
        if chart_data:
            response_data["chart"] = chart_data
        
        return QueryResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur query: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/questions")
async def get_questions(request: QuestionsRequest):
    """Génération de suggestions de questions"""
    try:
        session_id = request.session_id
        
        if session_id not in datasets:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        df = datasets[session_id]['dataframe']
        questions = generate_smart_questions(df)
        
        return {"session_id": session_id, "questions": questions}
        
    except Exception as e:
        print(f"Erreur questions: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Suppression d'une session"""
    if session_id in datasets:
        del datasets[session_id]
        if session_id in agents:
            del agents[session_id]
        return {"message": f"Session {session_id} supprimée"}
    else:
        raise HTTPException(status_code=404, detail="Session non trouvée")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
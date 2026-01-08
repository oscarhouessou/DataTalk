"""
API DataTalk - Version Minimaliste
Version sans LangChain pour éviter les conflits de dépendances
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import io
import os
from dotenv import load_dotenv
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Charger les variables d'environnement
load_dotenv()

# Configuration LLM Groq
def get_client_info():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        ), "llama-3.3-70b-versatile"
    else:
        print("❌ GROQ_API_KEY manquante dans .env")
        return None, None

client, default_model = get_client_info()

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

# Le montage des fichiers statiques sera fait APRES la définition des routes
# afin de garantir que les endpoints de l'API restent prioritaires.

# Stockage des datasets
datasets = {}

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    success: bool = True
    chart: str = None

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list
    success: bool = True

class QuestionsRequest(BaseModel):
    session_id: str

class ChartRequest(BaseModel):
    session_id: str
    chart_type: str = "auto"

def analyze_locally(df, query):
    """Analyse locale pour les questions simples"""
    query_lower = query.lower()
    
    # Questions sur la taille
    if any(word in query_lower for word in ['taille', 'lignes', 'colonnes', 'dimensions']):
        return f"Le dataset contient <strong>{len(df)} lignes</strong> et <strong>{len(df.columns)} colonnes</strong>.<br>Colonnes disponibles: {', '.join(df.columns.tolist())}"
    
    # Questions sur les statistiques
    if any(word in query_lower for word in ['statistique', 'résumé', 'summary', 'describe']):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats_text = "<strong>Statistiques des colonnes numériques:</strong><br><br>"
            for col in numeric_cols:
                stats = df[col].describe()
                stats_text += f"<strong>{col}:</strong><br>"
                stats_text += f"• Moyenne: {stats['mean']:.2f}<br>"
                stats_text += f"• Médiane: {stats['50%']:.2f}<br>"
                stats_text += f"• Écart-type: {stats['std']:.2f}<br>"
                stats_text += f"• Min: {stats['min']:.2f} | Max: {stats['max']:.2f}<br><br>"
            return stats_text
    
    # Questions sur les valeurs manquantes
    if any(word in query_lower for word in ['manquant', 'null', 'vide', 'missing']):
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            missing_by_col = df.isnull().sum()
            missing_details = "<strong>Valeurs manquantes par colonne:</strong><br>"
            for col, count in missing_by_col[missing_by_col > 0].items():
                missing_details += f"• {col}: {count} valeurs manquantes<br>"
            return f"Total: <strong>{missing_count} valeurs manquantes</strong><br><br>{missing_details}"
        else:
            return "✅ <strong>Aucune valeur manquante</strong> dans le dataset !"
    
    # Questions sur les colonnes
    if any(word in query_lower for word in ['colonnes', 'columns', 'variables']):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = f"<strong>Colonnes du dataset ({len(df.columns)} au total):</strong><br><br>"
        if numeric_cols:
            result += f"🔢 <strong>Numériques ({len(numeric_cols)}):</strong> {', '.join(numeric_cols)}<br><br>"
        if categorical_cols:
            result += f"📝 <strong>Catégorielles ({len(categorical_cols)}):</strong> {', '.join(categorical_cols)}"
        
        return result
    
    return None  # Utiliser OpenAI pour les questions complexes

def analyze_with_openai(df, query):
    """Analyse des données avec OpenAI directement"""
    try:
        if not client:
            return "❌ Erreur: GROQ_API_KEY non configurée."
        
        # Créer un résumé détaillé
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Statistiques numériques réelles
        numeric_stats = ""
        if numeric_cols:
            for col in numeric_cols:
                stats = df[col].describe()
                numeric_stats += f"""
  - {col}: 
    - Moyenne: {stats['mean']:.2f}
    - Médiane: {stats['50%']:.2f}
    - Écart-type: {stats['std']:.2f}
    - Min: {stats['min']:.2f}
    - Max: {stats['max']:.2f}"""
        
        # Informations catégorielles réelles
        categorical_info = ""
        if categorical_cols:
            for col in categorical_cols:
                unique_values = df[col].unique()[:10]  # Limiter à 10 valeurs
                categorical_info += f"""
  - {col}: {len(df[col].unique())} valeurs uniques - Exemples: {', '.join(map(str, unique_values))}"""
        
        summary = f"""
Dataset info:
- Lignes: {len(df)}
- Colonnes: {len(df.columns)}
- Colonnes: {', '.join(df.columns.tolist())}
- Valeurs manquantes: {df.isnull().sum().sum()} au total

Statistiques numériques:{numeric_stats}

Variables catégorielles:{categorical_info}

Échantillon des données:
{df.head(3).to_string()}
"""
        
        prompt = f"""
Tu es un analyste de données expert. Analyse le dataset suivant et réponds à la question de l'utilisateur.

{summary}

Question de l'utilisateur: {query}

Donne une réponse claire et précise basée sur les données disponibles.
"""

        response = client.chat.completions.create(
            model=default_model,
            messages=[
                {"role": "system", "content": "Tu es un expert en analyse de données."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Erreur OpenAI: {e}")
        return f"Analyse basique: Le dataset contient {len(df)} lignes et {len(df.columns)} colonnes."

def create_simple_chart(df, query):
    """Crée un graphique simple basé sur la requête"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        plt.figure(figsize=(10, 6))
        chart_created = False
        
        # Distribution des catégories
        if any(word in query.lower() for word in ['distribution', 'répartition']) and categorical_cols:
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'Distribution de {col}')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            chart_created = True
            
        # Histogramme pour les colonnes numériques
        elif any(word in query.lower() for word in ['moyenne', 'distribution']) and numeric_cols:
            plt.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7)
            plt.title(f'Distribution de {numeric_cols[0]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Fréquence')
            chart_created = True
        
        if chart_created:
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{chart_data}"
            
    except Exception as e:
        print(f"Erreur graphique: {e}")
    
    return None

@app.get("/", include_in_schema=False)
async def root():
    # Servir l'interface web si elle existe, sinon renvoyer un message API
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    return {"message": "DataTalk API Minimal", "version": "1.0.0", "status": "running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Format non supporté")
        
        df = df.dropna(how='all').reset_index(drop=True)
        
        datasets[session_id] = {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': pd.Timestamp.now()
        }
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        session_id = request.session_id
        query = request.query
        
        # Si pas de dataset, répondre de façon générique
        if session_id not in datasets:
            answer = "Je ne peux pas analyser de données car aucun fichier n'a été téléchargé. Veuillez d'abord télécharger un fichier CSV ou Excel pour que je puisse vous aider à analyser vos données."
            
            return QueryResponse(
                session_id=session_id,
                query=query,
                answer=answer,
                success=True
            )
        
        df = datasets[session_id]['dataframe']
        
        # Essayer d'abord l'analyse locale (plus rapide)
        answer = analyze_locally(df, query)
        
        # Si l'analyse locale ne peut pas répondre, utiliser OpenAI
        if answer is None:
            answer = analyze_with_openai(df, query)
        
        # Créer graphique si approprié
        chart_data = create_simple_chart(df, query)
        
        response = QueryResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            success=True
        )
        
        if chart_data:
            response.chart = chart_data
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/chat")
async def process_chat(request: QueryRequest):
    """Endpoint de compatibilité - redirige vers /query"""
    return await process_query(request)

@app.post("/questions")
async def get_questions(request: QuestionsRequest):
    try:
        session_id = request.session_id
        
        # Si pas de dataset, retourner des questions génériques
        if session_id not in datasets:
            questions = [
                "Comment puis-je commencer l'analyse de données ?",
                "Quels types de fichiers puis-je télécharger ?",
                "Peux-tu m'expliquer les fonctionnalités disponibles ?",
                "Comment interpréter les graphiques générés ?",
                "Quelles sont les meilleures pratiques d'analyse ?",
                "Comment exporter mes résultats d'analyse ?"
            ]
            return {"session_id": session_id, "questions": questions, "success": True}
        
        df = datasets[session_id]['dataframe']
        
        questions = [
            "Combien y a-t-il de lignes dans le dataset ?",
            f"Quelle est la distribution de {df.columns[0]} ?" if len(df.columns) > 0 else "Quelles sont les colonnes disponibles ?",
            f"Quelle est la moyenne de {df.select_dtypes(include=['number']).columns[0]} ?" if len(df.select_dtypes(include=['number']).columns) > 0 else "Y a-t-il des valeurs manquantes ?",
            "Quelles sont les valeurs uniques dans la première colonne ?",
            "Peux-tu me donner un résumé statistique des données ?",
            "Y a-t-il des doublons dans les données ?"
        ]
        
        return {"session_id": session_id, "questions": questions, "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/insights")
async def get_insights(request: QuestionsRequest):
    """Générer des insights automatiques sur le dataset"""
    try:
        session_id = request.session_id
        
        if session_id not in datasets:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        df = datasets[session_id]['dataframe']
        
        # Générer des insights basiques
        insights = []
        
        # Insight sur la taille
        insights.append(f"📊 <strong>Taille du dataset</strong>: {len(df)} lignes et {len(df.columns)} colonnes")
        
        # Insight sur les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            insights.append(f"🔢 <strong>Colonnes numériques</strong>: {len(numeric_cols)} colonnes ({', '.join(numeric_cols.tolist())})")
        
        # Insight sur les valeurs manquantes
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            insights.append(f"⚠️ <strong>Valeurs manquantes</strong>: {missing_count} valeurs manquantes au total")
        else:
            insights.append("✅ <strong>Qualité des données</strong>: Aucune valeur manquante détectée")
        
        # Insight sur les doublons
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            insights.append(f"🔄 <strong>Doublons</strong>: {duplicates} lignes dupliquées détectées")
        else:
            insights.append("✅ <strong>Unicité</strong>: Aucun doublon détecté")
        
        # Insights sur les colonnes catégorielles
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            insights.append(f"📝 <strong>Colonnes catégorielles</strong>: {len(categorical_cols)} colonnes ({', '.join(categorical_cols.tolist())})")
        
        # Formatage avec des séparateurs plus clairs
        insights_text = "\n\n".join(insights)
        
        # Formatage HTML avec plus d'espacement et meilleure séparation
        insights_html_formatted = []
        for i, insight in enumerate(insights):
            border_style = "" if i == len(insights) - 1 else "border-bottom: 1px solid #f0f0f0;"
            insights_html_formatted.append(f'<div style="margin-bottom: 20px; padding: 12px 0; {border_style} line-height: 1.6; font-size: 14px;">{insight}</div>')
        
        return {
            "session_id": session_id,
            "insights": insights_text,
            "insights_html": "".join(insights_html_formatted),
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/chart")
async def generate_chart(request: ChartRequest):
    """Génération de graphiques sur demande"""
    try:
        session_id = request.session_id
        chart_type = request.chart_type
        
        if session_id not in datasets:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        df = datasets[session_id]['dataframe']
        
        # Créer un graphique basé sur le type demandé
        chart_data = create_simple_chart(df, f"graphique {chart_type}")
        
        if chart_data:
            return {
                "session_id": session_id,
                "chart": chart_data,
                "success": True
            }
        else:
            return {
                "session_id": session_id,
                "message": "Aucun graphique généré",
                "success": False
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in datasets:
        del datasets[session_id]
        return {"message": f"Session {session_id} supprimée"}
    else:
        raise HTTPException(status_code=404, detail="Session non trouvée")

# Monter les fichiers statiques pour servir l'application web (doit être APRES les routes API)
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
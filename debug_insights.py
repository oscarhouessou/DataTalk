
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
import sys

# Load env vars
load_dotenv()

# Vérification de la clé API Groq
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("❌ GROQ_API_KEY non trouvée dans .env")
    exit(1)
else:
    print(f"Groq API Key present: {bool(groq_api_key)}")
    print(f"Groq API Key start: {groq_api_key[:5]}...")

# Mock Streamlit
class MockStreamlit:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
sys.modules['streamlit'] = MockStreamlit()

# Import nlq
try:
    import nlq
    print("nlq imported successfully")
except ImportError as e:
    print(f"Error importing nlq: {e}")
    sys.exit(1)

# Create dummy dataframe
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

# Initialize LLM
try:
    # Utiliser Groq uniquement
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile", 
        temperature=0.1,
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1"
    )
    print("🚀 Utilisation de Groq (llama-3.3-70b-versatile)")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    sys.exit(1)

# Run detect_automatic_insights
print("Running detect_automatic_insights...")
try:
    # We need to bypass the try-except block in nlq.py to see the error
    # But we can't easily do that without modifying the file.
    # However, if we call it, and it returns the error string, we know it failed.
    # To debug, we can try to run the code inside detect_automatic_insights manually here.
    
    # Replicate the logic from detect_automatic_insights
    stats_summary = df.describe().to_string()
    missing_data = df.isnull().sum()
    missing_info = missing_data[missing_data > 0].to_string() if missing_data.sum() > 0 else "Aucune donnée manquante"
    sample_data = df.head(5).to_string()
    
    prompt = f"""
    Analyse ce dataset et identifie 3-4 insights automatiques intéressants et actionables.
    
    Statistiques descriptives:
    {stats_summary}
    
    Données manquantes:
    {missing_info}
    
    Échantillon des données:
    {sample_data}
    
    Identifie des insights du type:
    - Anomalies ou valeurs surprenantes
    - Distributions intéressantes
    - Déséquilibres dans les données
    - Patterns ou tendances évidentes
    - Qualité des données
    
    Réponds avec 3-4 points courts et actionables, format:
    • Insight 1: Description courte et claire
    • Insight 2: Description courte et claire
    • Insight 3: Description courte et claire
    
    Sois concis et pratique.
    """
    
    print("Invoking LLM directly...")
    response = llm.invoke(prompt)
    print("LLM Response received:")
    print(response.content)
    
except Exception as e:
    print(f"Caught exception during manual execution: {e}")
    import traceback
    traceback.print_exc()

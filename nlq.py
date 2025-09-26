import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
import re

# -----------------------
# 🔹 Fonction pour demander un graphique intelligent à l'IA
# -----------------------
def get_chart_recommendation(llm, df, question, answer):
    """Demande à l'IA si un graphique serait utile et lequel"""
    
    # Obtenir les informations sur les colonnes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None, None
    
    # Créer le prompt pour l'IA avec des règles claires
    chart_prompt = f"""
    Basé sur cette question: "{question}"
    Et cette réponse: "{answer}"
    
    Colonnes disponibles:
    - Numériques: {numeric_cols}
    - Catégorielles: {categorical_cols}
    - Dates: {date_cols}
    
    RÈGLES IMPORTANTES pour décider si créer un graphique:
    
    NE PAS créer de graphique si:
    - La réponse est juste un nombre simple (ex: "Il y a 43 utilisateurs...")
    - C'est une statistique unique (moyenne, somme, total, pourcentage simple)
    - La question demande juste un comptage ou un calcul
    - La réponse est descriptive sans données à visualiser
    
    CRÉER un graphique seulement si:
    - Il y a une comparaison entre plusieurs catégories/groupes
    - Il y a une évolution dans le temps à montrer
    - Il y a une distribution/répartition à visualiser
    - Il y a des relations/corrélations entre variables
    - Cela aide vraiment à la compréhension des données
    
    Est-ce qu'un graphique aiderait VRAIMENT à mieux comprendre cette réponse?
    
    Si OUI, réponds EXACTEMENT dans ce format:
    GRAPHIQUE: OUI
    TYPE: [bar/line/scatter/hist/heatmap]
    COLONNES: [nom_colonne1, nom_colonne2]
    TITRE: [titre du graphique]
    
    Si NON, réponds simplement:
    GRAPHIQUE: NON
    
    Ne donne aucune autre explication, juste le format demandé.
    """
    
    try:
        chart_response = llm.invoke(chart_prompt)
        chart_decision = chart_response.content.strip()
        
        if "GRAPHIQUE: NON" in chart_decision:
            return None, None
            
        if "GRAPHIQUE: OUI" in chart_decision:
            return chart_decision, True
            
        return None, None
        
    except Exception as e:
        return None, None

# -----------------------
# 🔹 Fonction pour créer le graphique basé sur la recommandation IA
# -----------------------
def create_chart_from_ai_recommendation(df, recommendation):
    """Crée un graphique basé sur la recommandation de l'IA"""
    
    if not recommendation:
        return None
        
    try:
        lines = recommendation.split('\n')
        chart_type = None
        columns = []
        title = "Graphique"
        
        for line in lines:
            if line.startswith('TYPE:'):
                chart_type = line.replace('TYPE:', '').strip().lower()
            elif line.startswith('COLONNES:'):
                cols_str = line.replace('COLONNES:', '').strip()
                columns = [col.strip() for col in cols_str.replace('[', '').replace(']', '').split(',')]
            elif line.startswith('TITRE:'):
                title = line.replace('TITRE:', '').strip()
        
        if not chart_type or not columns:
            return None
        
        # Nettoyer les noms de colonnes
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'bar' and len(columns) >= 1:
            if len(columns) == 1:
                # Graphique en barres simple
                data = df[columns[0]].value_counts().head(10)
                data.plot(kind='bar', ax=ax, color='skyblue')
            else:
                # Graphique en barres groupé
                if df[columns[0]].dtype == 'object':
                    grouped = df.groupby(columns[0])[columns[1]].mean().head(10)
                    grouped.plot(kind='bar', ax=ax, color='skyblue')
                    
        elif chart_type == 'line' and len(columns) >= 1:
            if len(columns) == 1:
                df[columns[0]].plot(kind='line', ax=ax, marker='o')
            else:
                ax.plot(df[columns[0]], df[columns[1]], marker='o')
                
        elif chart_type == 'scatter' and len(columns) >= 2:
            ax.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            
        elif chart_type == 'hist' and len(columns) >= 1:
            ax.hist(df[columns[0]].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax.set_xlabel(columns[0])
            ax.set_ylabel('Fréquence')
            
        elif chart_type == 'heatmap' and len(columns) >= 2:
            # Matrice de corrélation
            numeric_data = df[columns].select_dtypes(include=['number'])
            if len(numeric_data.columns) >= 2:
                corr_matrix = numeric_data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.warning(f"Erreur lors de la création du graphique : {str(e)}")
        return None

# -----------------------
# ⚙️ Configuration Streamlit
# -----------------------
st.set_page_config(page_title="NLQ Data Analyst MVP", layout="wide")
st.title("🤖 NLQ Data Analyst MVP")
st.markdown(
    "Téléversez un fichier CSV ou Excel et posez vos questions en langage naturel."
)

# -----------------------
# 🔑 Chargement des variables d'environnement
# -----------------------
load_dotenv()

# Vérification de la clé API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your_openai_api_key_here":
    st.error("🔑 Clé API OpenAI non configurée !")
    st.markdown("""
    **Pour configurer votre clé API :**
    1. Ouvrez le fichier `.env` dans ce dossier
    2. Remplacez `your_openai_api_key_here` par votre vraie clé API OpenAI
    3. Sauvegardez le fichier et rechargez l'application
    
    **Obtenir une clé API :** https://platform.openai.com/account/api-keys
    """)
    st.stop()

# -----------------------
# 🔹 Upload du fichier
# -----------------------
uploaded_file = st.file_uploader(
    "Choisir un fichier CSV ou Excel", type=["csv", "xlsx"]
)

if uploaded_file:
    # Lecture du fichier
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Aperçu des données :")
    st.dataframe(df.head())

    # -----------------------
    # 🔹 Initialisation du chat et de l'agent
    # -----------------------
    
    # Initialiser l'historique du chat dans la session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Créer l'agent LangChain une seule fois
    if "agent" not in st.session_state:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Un peu de créativité pour la conversation
        
        # Prompt système pour rendre l'agent plus conversationnel
        system_message = """
        Tu es un expert analyste de données très amical et conversationnel. 
        
        Instructions importantes:
        - Réponds de manière naturelle et engageante, comme dans une vraie conversation
        - Explique tes analyses de façon claire et accessible
        - N'hésite pas à poser des questions de clarification si nécessaire
        - Propose des insights intéressants et des observations pertinentes
        - Sois proactif en suggérant d'autres analyses qui pourraient être utiles
        - Utilise un ton professionnel mais décontracté
        - Si tu identifies des tendances intéressantes, mentionne-les
        - Contextualise tes réponses par rapport aux données analysées
        
        Pour les questions statistiques simples (nombres, moyennes, sommes):
        - Donne la réponse directe et claire
        - Ajoute du contexte et des insights pertinents
        - Suggère des analyses plus approfondies qui pourraient nécessiter des visualisations
        
        Quand tu réponds, structure ta réponse ainsi:
        1. Réponse directe à la question
        2. Analyse et insights (contextualisation, implications)
        3. Suggestions pour approfondir (optionnel - propose des analyses qui bénéficieraient de visualisations)
        """
        
        st.session_state.agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, agent_type="openai-functions", 
            allow_dangerous_code=True,
            prefix=system_message
        )
        st.session_state.llm = llm  # Garder une référence au LLM pour les graphiques

    # -----------------------
    # 🔹 Interface de Chat
    # -----------------------
    st.write("### 💬 Chat avec vos données")
    
    # Afficher l'historique du chat
    for i, (question, answer, chart_info) in enumerate(st.session_state.chat_history):
        # Question de l'utilisateur
        with st.chat_message("user"):
            st.write(question)
        
        # Réponse de l'assistant
        with st.chat_message("assistant"):
            st.write(answer)
            
            # Afficher le graphique s'il existe
            if chart_info:
                st.pyplot(chart_info)

    # -----------------------
    # 🔹 Nouvelle question
    # -----------------------
    if prompt := st.chat_input("Posez votre question sur les données..."):
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.write(prompt)
        
        # Traitement et réponse
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Obtenir la réponse de l'agent
                    response = st.session_state.agent.invoke(prompt)
                    answer = response["output"]
                    
                    # Afficher la réponse
                    st.write(answer)
                    
                    # Demander à l'IA si un graphique serait utile
                    chart_recommendation, needs_chart = get_chart_recommendation(st.session_state.llm, df, prompt, answer)
                    
                    chart = None
                    if needs_chart:
                        chart = create_chart_from_ai_recommendation(df, chart_recommendation)
                        if chart:
                            st.pyplot(chart)
                    
                    # Ajouter à l'historique
                    st.session_state.chat_history.append((prompt, answer, chart))
                    
                except Exception as e:
                    error_msg = f"Erreur lors de l'analyse : {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append((prompt, error_msg, None))

    # -----------------------
    # 🔹 Bouton pour effacer l'historique
    # -----------------------
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.chat_history = []
        st.rerun()



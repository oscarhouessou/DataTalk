import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
import numpy as np

# -----------------------
# 🔹 Fonction pour générer des suggestions de questions intelligentes
# -----------------------
def generate_smart_questions(llm, df):
    """Génère des suggestions de questions basées sur l'analyse des données"""
    
    # Analyser la structure des données
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Échantillon des données pour l'analyse
    sample_data = df.head(3).to_string()
    
    prompt = f"""
    Analyse ce dataset et suggère 6 questions pertinentes et intéressantes qu'un analyste pourrait poser.
    
    Structure des données:
    - Colonnes numériques: {numeric_cols}
    - Colonnes catégorielles: {categorical_cols}
    - Colonnes dates: {date_cols}
    - Nombre de lignes: {len(df)}
    
    Échantillon des données:
    {sample_data}
    
    Génère 6 questions variées et pertinentes:
    - 2 questions d'exploration générale (distribution, aperçu)
    - 2 questions de comparaison/segmentation
    - 2 questions d'analyse approfondie (corrélations, tendances)
    
    Format: retourne uniquement une liste Python de strings, rien d'autre.
    Exemple: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?", "Question 6?"]
    """
    
    try:
        response = llm.invoke(prompt)
        suggestions_text = response.content.strip()
        
        # Extraire la liste des questions
        import ast
        try:
            questions = ast.literal_eval(suggestions_text)
            return questions if isinstance(questions, list) else []
        except Exception:
            # Fallback: extraire manuellement
            lines = suggestions_text.split('\n')
            questions = []
            for line in lines:
                if '?' in line:
                    # Nettoyer la ligne
                    clean_question = line.strip().strip('"\'\'').strip('- ').strip('* ')
                    if len(clean_question) > 10:
                        questions.append(clean_question)
            return questions[:6]
    except Exception:
        return []

# -----------------------
# 🔹 Fonction pour détecter des insights automatiques
# -----------------------
def detect_automatic_insights(llm, df):
    """Détecte automatiquement des insights intéressants dans les données"""
    
    # Statistiques de base
    stats_summary = df.describe().to_string()
    missing_data = df.isnull().sum()
    missing_info = missing_data[missing_data > 0].to_string() if missing_data.sum() > 0 else "Aucune donnée manquante"
    
    # Quelques échantillons
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
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Impossible de générer des insights automatiques pour le moment."

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
    
    Colonnes disponibles EXACTEMENT (utilise ces noms précis):
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
    - Il y a une distribution/répartition à visualiser (ex: "distribution des taux")
    - Il y a une évolution dans le temps à montrer
    - Il y a des relations/corrélations entre variables
    - Cela aide vraiment à la compréhension des données
    
    Types de graphiques recommandés (sois créatif et choisis le plus approprié):
    - bar: comparaisons entre catégories, classements
    - boxplot: distribution d'une variable numérique par catégories, détection d'outliers
    - violin: distribution plus détaillée que boxplot
    - hist: distribution d'une seule variable numérique
    - line: évolution temporelle, tendances
    - scatter: relation entre deux variables numériques
    - heatmap: corrélations multiples, matrices
    - pie: proportions (max 6-8 catégories)
    - area: évolution de volumes dans le temps
    - density: distribution continue lissée
    - pair: relations multiples entre plusieurs variables
    
    Si un graphique est utile, utilise les NOMS EXACTS des colonnes listées ci-dessus.
    
    Si OUI, réponds EXACTEMENT dans ce format:
    GRAPHIQUE: OUI
    TYPE: [bar/boxplot/line/scatter/hist/heatmap]
    COLONNES: [nom_exact_colonne1, nom_exact_colonne2]
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
        
    except Exception:
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
        
        # Nettoyer les noms de colonnes et vérifier qu'elles existent
        valid_columns = []
        for col in columns:
            # Essayer de trouver la colonne même avec des variations de nom
            col_clean = col.strip('"\' ')
            if col_clean in df.columns:
                valid_columns.append(col_clean)
            else:
                # Chercher une correspondance approximative
                for df_col in df.columns:
                    if col_clean.lower() in df_col.lower() or df_col.lower() in col_clean.lower():
                        valid_columns.append(df_col)
                        break
        
        if not valid_columns:
            return None
        
        # Créer le graphique avec un style plus robuste
        plt.style.use('default')  # Assurer un style cohérent
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('white')
        
        try:
            # Variables pour vérifier si le graphique a du contenu
            plot_created = False
            
            if chart_type == 'bar' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if len(valid_columns) == 1:
                    # Graphique en barres simple pour une colonne
                    if df[col1].dtype in ['object', 'category']:
                        data = df[col1].value_counts().head(15)
                        bars = ax.bar(range(len(data)), data.values, color='skyblue', edgecolor='navy', alpha=0.7)
                        ax.set_xticks(range(len(data)))
                        ax.set_xticklabels(data.index, rotation=45, ha='right')
                        ax.set_ylabel('Fréquence')
                        # Ajouter les valeurs sur les barres
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{int(height)}', ha='center', va='bottom')
                        plot_created = True
                    else:
                        # Histogramme pour données numériques
                        data_clean = df[col1].dropna()
                        if len(data_clean) > 0:
                            ax.hist(data_clean, bins=min(30, len(data_clean.unique())), 
                                   alpha=0.7, color='lightblue', edgecolor='navy')
                            ax.set_xlabel(col1)
                            ax.set_ylabel('Fréquence')
                            plot_created = True
                else:
                    # Graphique groupé pour deux colonnes
                    col2 = valid_columns[1]
                    if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
                        grouped = df.groupby(col1)[col2].mean().sort_values(ascending=False).head(15)
                        bars = ax.bar(range(len(grouped)), grouped.values, color='lightcoral', alpha=0.7)
                        ax.set_xticks(range(len(grouped)))
                        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                        ax.set_ylabel(f'Moyenne de {col2}')
                        ax.set_xlabel(col1)
                        plot_created = True
            
            elif chart_type == 'boxplot' and len(valid_columns) >= 2:
                col1, col2 = valid_columns[0], valid_columns[1]
                if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
                    # Créer un boxplot
                    groups = [group[col2].dropna() for name, group in df.groupby(col1)]
                    labels = [name for name, group in df.groupby(col1)]
                    ax.boxplot(groups, labels=labels)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    plot_created = True
            
            elif chart_type == 'violin' and len(valid_columns) >= 2:
                col1, col2 = valid_columns[0], valid_columns[1]
                if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
                    # Créer un violin plot
                    data_for_violin = [group[col2].dropna() for name, group in df.groupby(col1) if len(group) > 5]
                    labels_for_violin = [name for name, group in df.groupby(col1) if len(group) > 5]
                    if len(data_for_violin) > 0:
                        ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)))
                        ax.set_xticks(range(len(labels_for_violin)))
                        ax.set_xticklabels(labels_for_violin, rotation=45, ha='right')
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        plot_created = True
            
            elif chart_type == 'pie' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['object', 'category']:
                    data = df[col1].value_counts().head(8)  # Max 8 segments
                    if len(data) > 0:
                        colors = plt.cm.Set3(range(len(data)))
                        wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                                                          colors=colors, startangle=90)
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        plot_created = True
            
            elif chart_type == 'area' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['int64', 'float64']:
                    ax.fill_between(df.index, df[col1], alpha=0.7, color='lightblue')
                    ax.plot(df.index, df[col1], color='navy', linewidth=2)
                    ax.set_ylabel(col1)
                    ax.set_xlabel('Index')
                    plot_created = True
            
            elif chart_type == 'density' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['int64', 'float64']:
                    data_clean = df[col1].dropna()
                    if len(data_clean) > 10:
                        data_clean.plot.density(ax=ax, color='purple', linewidth=2)
                        ax.set_xlabel(col1)
                        ax.set_ylabel('Densité')
                        ax.fill_between(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata(), alpha=0.3, color='purple')
                        plot_created = True
            
            elif chart_type == 'line' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if len(valid_columns) == 1:
                    ax.plot(df.index, df[col1], marker='o', linewidth=2, markersize=4)
                    ax.set_ylabel(col1)
                    ax.set_xlabel('Index')
                else:
                    col2 = valid_columns[1]
                    ax.plot(df[col1], df[col2], marker='o', linewidth=2, markersize=4)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                plot_created = True
            
            elif chart_type == 'scatter' and len(valid_columns) >= 2:
                col1, col2 = valid_columns[0], valid_columns[1]
                if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                    ax.scatter(df[col1], df[col2], alpha=0.6, s=50, color='coral')
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    plot_created = True
            
            elif chart_type == 'hist' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['int64', 'float64']:
                    data_clean = df[col1].dropna()
                    if len(data_clean) > 0:
                        ax.hist(data_clean, bins=min(30, len(data_clean.unique())), 
                               alpha=0.7, color='lightgreen', edgecolor='darkgreen')
                        ax.set_xlabel(col1)
                        ax.set_ylabel('Fréquence')
                        plot_created = True
            
                    plot_created = True
            
            elif chart_type == 'pie' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['object', 'category']:
                    value_counts = df[col1].value_counts().head(6)  # Max 6 pour lisibilité
                    if len(value_counts) > 0:
                        colors = plt.cm.Set3(range(len(value_counts)))
                        wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                                          autopct='%1.1f%%', colors=colors, startangle=90)
                        ax.set_title(f'Répartition de {col1}', fontsize=14, fontweight='bold')
                        chart_created = True
            
            elif chart_type == 'violin' and len(valid_columns) >= 2:
                col1, col2 = valid_columns[0], valid_columns[1]
                if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
                    # Créer un violin plot
                    data_for_violin = [group[col2].dropna() for name, group in df.groupby(col1) if len(group[col2].dropna()) > 0]
                    labels_for_violin = [name for name, group in df.groupby(col1) if len(group[col2].dropna()) > 0]
                    if data_for_violin:
                        ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)), showmeans=True)
                        ax.set_xticks(range(len(labels_for_violin)))
                        ax.set_xticklabels(labels_for_violin, rotation=45, ha='right')
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        chart_created = True
            
            elif chart_type == 'density' and len(valid_columns) >= 1:
                col1 = valid_columns[0]
                if df[col1].dtype in ['int64', 'float64']:
                    data_clean = df[col1].dropna()
                    if len(data_clean) > 1:
                        ax.hist(data_clean, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='navy')
                        # Ajouter une courbe de densité
                        try:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(data_clean)
                            x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Densité')
                            ax.legend()
                        except ImportError:
                            pass  # scipy pas disponible, juste l'histogramme
                        ax.set_xlabel(col1)
                        ax.set_ylabel('Densité')
                        chart_created = True
            
            elif chart_type == 'area' and len(valid_columns) >= 2:
                col1, col2 = valid_columns[0], valid_columns[1]
                if df[col2].dtype in ['int64', 'float64']:
                    # Trier par la première colonne si possible
                    df_sorted = df.sort_values(col1) if df[col1].dtype in ['int64', 'float64'] else df
                    ax.fill_between(range(len(df_sorted)), df_sorted[col2], alpha=0.6, color='lightgreen')
                    ax.plot(range(len(df_sorted)), df_sorted[col2], color='darkgreen', linewidth=2)
                    ax.set_xlabel('Index' if df[col1].dtype not in ['int64', 'float64'] else col1)
                    ax.set_ylabel(col2)
                    chart_created = True
            
            elif chart_type == 'heatmap' and len(valid_columns) >= 2:
                # Matrice de corrélation
                numeric_cols = [col for col in valid_columns if df[col].dtype in ['int64', 'float64']]
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                               square=True, fmt='.2f')
                    chart_created = True
                    plot_created = True
            
            # Améliorer l'apparence du graphique seulement si un plot a été créé
            if plot_created:
                if chart_type != 'pie':  # Le pie chart gère son propre titre
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                if chart_type not in ['pie', 'heatmap']:  # Pas de grille pour pie et heatmap
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return fig
            else:
                # Aucun graphique n'a été créé avec succès
                plt.close(fig)
                return None
            
        except Exception as plot_error:
            plt.close(fig)
            st.warning(f"Erreur lors de la création du graphique spécifique : {str(plot_error)}")
            return None
        
    except Exception as e:
        st.warning(f"Erreur lors de l'analyse de la recommandation : {str(e)}")
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

    st.write("### 📊 Aperçu des données :")
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
        
        # Prompt système pour un data analyst expert
        system_message = """
        Tu es un DATA ANALYST EXPERT de niveau senior avec 10+ années d'expérience dans l'analyse de données complexes.
        Tu maîtrises parfaitement pandas, les statistiques avancées, et l'interprétation de données business.
        
        EXPERTISE ET COMPÉTENCES:
        - Analyse statistique approfondie (corrélations, distributions, tests statistiques)
        - Détection d'anomalies et patterns cachés
        - Segmentation intelligente des données
        - Interprétation business des résultats
        - Recommandations stratégiques basées sur les données
        
        RÈGLES ABSOLUES:
        - JAMAIS de réponses génériques ou hypothétiques
        - TOUJOURS exécuter du code pandas pour obtenir des résultats RÉELS
        - TOUJOURS fournir des statistiques précises et des insights métier
        - OBLIGATION d'analyser les vraies données du dataset fourni
        
        MÉTHODOLOGIE D'ANALYSE:
        1. Exploration des données (shape, types, valeurs manquantes)
        2. Calculs statistiques précis (moyennes, médianes, écarts-types, quartiles)
        3. Détection d'outliers et anomalies
        4. Segmentation et comparaisons pertinentes
        5. Insights business et recommandations concrètes
        
        COMMUNICATION:
        - Ton professionnel d'expert mais accessible
        - Explications claires avec vulgarisation si nécessaire
        - Toujours contextualiser les résultats par rapport au business
        - Proposer des analyses complémentaires pertinentes
        - Alerter sur les limitations ou biais potentiels des données
        
        STRUCTURE DE RÉPONSE OBLIGATOIRE:
        1. **Analyse Quantitative**: Chiffres précis et statistiques clés
        2. **Interprétation Expert**: Signification business des résultats
        3. **Insights Avancés**: Patterns, anomalies, corrélations découvertes
        4. **Recommandations**: Actions concrètes basées sur les findings
        5. **Analyses Complémentaires**: Suggestions d'approfondissement
        
        En tant qu'expert, tu dois toujours aller au-delà des statistiques de base pour révéler des insights métier actionables.
        """
        
        st.session_state.agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, agent_type="openai-functions", 
            allow_dangerous_code=True,
            prefix=system_message
        )
        st.session_state.llm = llm  # Garder une référence au LLM pour les graphiques

    # -----------------------
    # 🔹 Sidebar avec insights automatiques et suggestions
    # -----------------------
    with st.sidebar:
        st.header("🧠 Analyse Intelligente")
        
        # Insights automatiques
        with st.expander("🔍 Insights Automatiques", expanded=True):
            with st.spinner("Génération d'insights..."):
                if "auto_insights" not in st.session_state:
                    st.session_state.auto_insights = detect_automatic_insights(st.session_state.llm, df)
                st.markdown(st.session_state.auto_insights)
        
        # Suggestions de questions
        with st.expander("💡 Questions Suggérées", expanded=True):
            with st.spinner("Génération de suggestions..."):
                if "suggested_questions" not in st.session_state:
                    st.session_state.suggested_questions = generate_smart_questions(st.session_state.llm, df)
                
                if st.session_state.suggested_questions:
                    st.markdown("**Cliquez sur une question pour l'utiliser :**")
                    for i, question in enumerate(st.session_state.suggested_questions):
                        if st.button(f"❓ {question}", key=f"suggested_{i}", use_container_width=True):
                            # Ajouter la question au chat
                            st.session_state.pending_question = question
                            st.rerun()
                else:
                    st.info("Aucune suggestion disponible pour le moment.")
        
        # Bouton pour régénérer les analyses
        if st.button("🔄 Régénérer les analyses", use_container_width=True):
            if "auto_insights" in st.session_state:
                del st.session_state.auto_insights
            if "suggested_questions" in st.session_state:
                del st.session_state.suggested_questions
            st.rerun()

    # -----------------------
    # 🔹 Interface de Chat
    # -----------------------
    st.write("### 💬 Chat avec vos données")
    
    # Traiter une question suggérée en attente
    if "pending_question" in st.session_state:
        pending_q = st.session_state.pending_question
        del st.session_state.pending_question
        
        # Traiter la question suggérée directement et l'ajouter à l'historique
        try:
            response = st.session_state.agent.invoke(pending_q)
            answer = response["output"]
            
            # Vérifier si un graphique est nécessaire
            chart_recommendation, needs_chart = get_chart_recommendation(st.session_state.llm, df, pending_q, answer)
            
            chart = None
            if needs_chart:
                chart = create_chart_from_ai_recommendation(df, chart_recommendation)
            
            # Ajouter à l'historique SEULEMENT (pas d'affichage direct)
            st.session_state.chat_history.append((pending_q, answer, chart))
            
        except Exception as e:
            error_msg = f"Erreur lors de l'analyse : {str(e)}"
            st.session_state.chat_history.append((pending_q, error_msg, None))
        
        # Forcer le rechargement pour afficher la nouvelle entrée dans l'historique
        st.rerun()
    
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
    # 🔹 Boutons de contrôle
    # -----------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 Nouveau fichier", use_container_width=True):
            # Réinitialiser toute la session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()



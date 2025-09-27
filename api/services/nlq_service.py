"""
Service NLQ - Traitement des requêtes en langage naturel
Réutilise la logique du nlq.py sans le modifier
"""

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
import ast
from typing import Dict, List, Any, Optional
import asyncio
import logging

from ..models.api_models import QueryResponse, QuestionsResponse
from ..config.settings import get_settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NLQService:
    """Service pour le traitement de requêtes en langage naturel"""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    async def process_query(
        self, 
        query: str, 
        dataframe: pd.DataFrame, 
        chat_history: List[Dict] = None
    ) -> QueryResponse:
        """
        Traite une requête en langage naturel sur les données
        Réutilise la logique de l'agent pandas du nlq.py
        """
        try:
            # Créer l'agent pandas avec le même système de prompt que nlq.py
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=dataframe,
                verbose=True,
                allow_dangerous_code=True,
                agent_executor_kwargs={
                    'handle_parsing_errors': True,
                    'max_iterations': 3
                },
                prefix="""
                Tu es un expert en analyse de données et un data scientist senior avec plus de 10 ans d'expérience.
                Tu travailles avec un DataFrame pandas pour répondre aux questions business de manière professionnelle.
                
                INSTRUCTIONS IMPORTANTES:
                
                1. 📊 ANALYSE BUSINESS : Adopte toujours une perspective d'analyste senior qui comprend les enjeux business
                2. 🎯 RÉPONSES PRÉCISES : Sois factuel, précis et actionnable dans tes réponses
                3. 💡 INSIGHTS STRATÉGIQUES : Va au-delà des chiffres, explique ce qu'ils signifient pour le business
                4. 📈 COMPARAISONS INTELLIGENTES : Utilise des références et des contextes pertinents
                5. ⚡ EFFICACITÉ : Code propre et optimisé, pas de solutions compliquées pour des problèmes simples
                
                FORMAT DE RÉPONSE OBLIGATOIRE:
                - Commence par un résumé exécutif clair
                - Donne les chiffres clés avec leur contexte business
                - Ajoute 2-3 insights stratégiques
                - Propose des recommandations d'actions si pertinent
                
                Tu as accès à ces outils:
                """
            )
            
            # Exécuter la requête
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: agent.invoke({"input": query})
            )
            
            # Extraire la réponse
            answer = result.get("output", "")
            
            # Vérifier s'il y a du code exécuté
            code_executed = None
            if hasattr(result, 'intermediate_steps') and result.intermediate_steps:
                for step in result.intermediate_steps:
                    if hasattr(step[0], 'tool') and step[0].tool == 'python_repl_ast':
                        code_executed = step[0].tool_input
                        break
            
            return QueryResponse(
                session_id="default",
                query=query,
                answer=answer,
                code_executed=code_executed,
                has_chart=False,
                execution_time=None
            )
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
            return QueryResponse(
                session_id="default",
                query=query,
                answer=f"Erreur lors du traitement de votre requête: {str(e)}",
                code_executed=None,
                has_chart=False,
                execution_time=None,
                success=False
            )
    
    async def generate_smart_questions(
        self, 
        dataframe: pd.DataFrame, 
        context: str = None
    ) -> List[Dict[str, str]]:
        """
        Génère des suggestions de questions intelligentes
        Réutilise la fonction generate_smart_questions du nlq.py
        """
        try:
            # Analyser la structure des données
            numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = dataframe.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Échantillon des données pour l'analyse
            sample_data = dataframe.head(3).to_string()
            
            prompt = f"""
            Analyse ce dataset et suggère 6 questions pertinentes et intéressantes qu'un analyste pourrait poser.
            
            Structure des données:
            - Colonnes numériques: {numeric_cols}
            - Colonnes catégorielles: {categorical_cols}
            - Colonnes dates: {date_cols}
            - Nombre de lignes: {len(dataframe)}
            
            Échantillon des données:
            {sample_data}
            
            {f"Contexte additionnel: {context}" if context else ""}
            
            Génère 6 questions variées et pertinentes:
            - 2 questions d'exploration générale (distribution, aperçu)
            - 2 questions de comparaison/segmentation
            - 2 questions d'analyse approfondie (corrélations, tendances)
            
            Format: retourne uniquement une liste Python de dictionnaires avec 'question' et 'category'.
            Exemple: [
                {"question": "Question 1?", "category": "exploration"},
                {"question": "Question 2?", "category": "comparaison"},
                ...
            ]
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            suggestions_text = response.content.strip()
            
            # Extraire la liste des questions
            try:
                questions = ast.literal_eval(suggestions_text)
                if isinstance(questions, list) and all(isinstance(q, dict) for q in questions):
                    return questions
            except Exception:
                # Fallback: extraire manuellement
                lines = suggestions_text.split('\n')
                questions = []
                categories = ["exploration", "comparaison", "analyse", "insights", "tendances", "distribution"]
                cat_index = 0
                
                for line in lines:
                    if '?' in line:
                        # Nettoyer la ligne
                        clean_question = line.strip().strip('"\'').strip('- ').strip('* ')
                        if len(clean_question) > 10:
                            questions.append({
                                "question": clean_question,
                                "category": categories[cat_index % len(categories)]
                            })
                            cat_index += 1
                            
                return questions[:6]
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des questions: {str(e)}")
            return [
                {"question": "Quelles sont les statistiques principales de ce dataset?", "category": "exploration"},
                {"question": "Y a-t-il des valeurs aberrantes dans les données?", "category": "qualité"},
                {"question": "Comment les données sont-elles distribuées?", "category": "distribution"}
            ]
    
    def get_chart_recommendation(
        self, 
        dataframe: pd.DataFrame, 
        question: str, 
        answer: str
    ) -> Optional[str]:
        """
        Demande à l'IA si un graphique serait utile et lequel
        Réutilise la fonction get_chart_recommendation du nlq.py
        """
        try:
            # Obtenir les informations sur les colonnes
            numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = dataframe.select_dtypes(include=['datetime64']).columns.tolist()
            
            if len(numeric_cols) == 0:
                return None
            
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
            
            Types de graphiques recommandés:
            - bar: comparaisons entre catégories, classements
            - boxplot: distribution d'une variable numérique par catégories
            - violin: distribution plus détaillée que boxplot
            - hist: distribution d'une seule variable numérique
            - line: évolution temporelle, tendances
            - scatter: relation entre deux variables numériques
            - heatmap: corrélations multiples, matrices
            - pie: proportions (max 6-8 catégories)
            - area: évolution de volumes dans le temps
            - density: distribution continue lissée
            
            Si un graphique est utile, utilise les NOMS EXACTS des colonnes listées ci-dessus.
            
            Si OUI, réponds EXACTEMENT dans ce format:
            GRAPHIQUE: OUI
            TYPE: [bar/boxplot/line/scatter/hist/heatmap]
            COLONNES: [nom_exact_colonne1, nom_exact_colonne2]
            TITRE: [titre du graphique]
            
            Si NON, réponds simplement:
            GRAPHIQUE: NON
            """
            
            chart_response = self.llm.invoke(chart_prompt)
            chart_decision = chart_response.content.strip()
            
            if "GRAPHIQUE: NON" in chart_decision:
                return None
                
            if "GRAPHIQUE: OUI" in chart_decision:
                return chart_decision
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la recommandation de graphique: {str(e)}")
            return None
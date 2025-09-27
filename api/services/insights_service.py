"""
Service d'analyse des insights automatiques
Réutilise la logique du nlq.py pour générer des insights intelligents
"""

import pandas as pd
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Optional
import logging

from ..models.api_models import InsightsResponse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class InsightsService:
    """Service pour la génération d'insights automatiques"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    async def detect_automatic_insights(
        self, 
        dataframe: pd.DataFrame,
        analysis_type: str = "comprehensive"
    ) -> List[Dict[str, str]]:
        """
        Détecte automatiquement des insights intéressants dans les données
        Réutilise la fonction detect_automatic_insights du nlq.py
        """
        try:
            # Statistiques de base
            stats_summary = dataframe.describe().to_string()
            missing_data = dataframe.isnull().sum()
            missing_info = missing_data[missing_data > 0].to_string() if missing_data.sum() > 0 else "Aucune donnée manquante"
            
            # Quelques échantillons
            sample_data = dataframe.head(5).to_string()
            
            # Informations sur les colonnes
            numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = dataframe.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Personnaliser le prompt selon le type d'analyse
            analysis_prompts = {
                "comprehensive": self._get_comprehensive_prompt(),
                "statistical": self._get_statistical_prompt(),
                "patterns": self._get_patterns_prompt(),
                "quality": self._get_quality_prompt()
            }
            
            base_prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
            
            prompt = f"""
            {base_prompt}
            
            Informations sur le dataset:
            - Nombre de lignes: {len(dataframe)}
            - Nombre de colonnes: {len(dataframe.columns)}
            - Colonnes numériques: {numeric_cols}
            - Colonnes catégorielles: {categorical_cols}
            - Colonnes dates: {date_cols}
            
            Statistiques descriptives:
            {stats_summary}
            
            Données manquantes:
            {missing_info}
            
            Échantillon des données:
            {sample_data}
            
            Génère 4-6 insights structurés au format JSON:
            [
                {{
                    "type": "anomalie|distribution|qualite|correlation|tendance",
                    "title": "Titre court de l'insight",
                    "description": "Description détaillée et actionnable",
                    "severity": "low|medium|high",
                    "recommendation": "Action recommandée"
                }},
                ...
            ]
            
            Retourne uniquement le JSON, rien d'autre.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            insights_text = response.content.strip()
            
            # Parser le JSON des insights
            import json
            try:
                insights = json.loads(insights_text)
                if isinstance(insights, list):
                    return insights
            except json.JSONDecodeError:
                # Fallback: parser manuellement
                return self._parse_insights_fallback(insights_text)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des insights: {str(e)}")
            return self._get_default_insights(dataframe)
    
    def _get_comprehensive_prompt(self) -> str:
        """Prompt pour une analyse complète"""
        return """
        Analyse ce dataset de manière complète et identifie des insights actionables.
        Focus sur:
        - Anomalies ou valeurs surprenantes
        - Distributions intéressantes ou déséquilibrées
        - Qualité des données (valeurs manquantes, outliers)
        - Patterns évidents dans les données
        - Recommandations d'actions business
        """
    
    def _get_statistical_prompt(self) -> str:
        """Prompt pour une analyse statistique"""
        return """
        Analyse statistique approfondie de ce dataset.
        Focus sur:
        - Distributions statistiques (normalité, skewness, etc.)
        - Corrélations significatives
        - Outliers statistiques
        - Variances et écarts-types inhabituels
        - Tests de significativité implicites
        """
    
    def _get_patterns_prompt(self) -> str:
        """Prompt pour la détection de patterns"""
        return """
        Détecte des patterns et tendances dans ce dataset.
        Focus sur:
        - Tendances temporelles si applicables
        - Groupements naturels dans les données
        - Cycles ou saisonnalités
        - Segments de population distincts
        - Relations entre variables
        """
    
    def _get_quality_prompt(self) -> str:
        """Prompt pour l'analyse de qualité des données"""
        return """
        Évalue la qualité de ce dataset.
        Focus sur:
        - Complétude des données (valeurs manquantes)
        - Cohérence des formats et types
        - Doublons potentiels
        - Valeurs aberrantes
        - Intégrité référentielle
        """
    
    def _parse_insights_fallback(self, insights_text: str) -> List[Dict[str, str]]:
        """Parser de fallback si le JSON échoue"""
        insights = []
        lines = insights_text.split('\n')
        
        current_insight = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('•') or line.startswith('-'):
                if current_insight:
                    insights.append(current_insight)
                current_insight = {
                    "type": "general",
                    "title": line.strip('•- '),
                    "description": line.strip('•- '),
                    "severity": "medium",
                    "recommendation": "Analyser plus en détail"
                }
            elif ':' in line and current_insight:
                key, value = line.split(':', 1)
                if key.lower().strip() in ['description', 'recommendation']:
                    current_insight[key.lower().strip()] = value.strip()
        
        if current_insight:
            insights.append(current_insight)
            
        return insights[:6]  # Max 6 insights
    
    def _get_default_insights(self, dataframe: pd.DataFrame) -> List[Dict[str, str]]:
        """Insights par défaut en cas d'erreur"""
        insights = []
        
        # Insight sur la taille
        insights.append({
            "type": "general",
            "title": f"Dataset de {len(dataframe)} lignes et {len(dataframe.columns)} colonnes",
            "description": f"Le dataset contient {len(dataframe):,} enregistrements avec {len(dataframe.columns)} variables à analyser.",
            "severity": "low",
            "recommendation": "Commencer par une analyse exploratoire des variables principales"
        })
        
        # Insight sur les valeurs manquantes
        missing_count = dataframe.isnull().sum().sum()
        if missing_count > 0:
            missing_pct = (missing_count / (len(dataframe) * len(dataframe.columns))) * 100
            insights.append({
                "type": "qualite",
                "title": f"{missing_count} valeurs manquantes détectées ({missing_pct:.1f}%)",
                "description": f"Le dataset contient {missing_count} valeurs manquantes, soit {missing_pct:.1f}% de l'ensemble des données.",
                "severity": "medium" if missing_pct > 10 else "low",
                "recommendation": "Analyser les patterns de valeurs manquantes et décider de la stratégie de traitement"
            })
        
        # Insight sur les types de données
        numeric_count = len(dataframe.select_dtypes(include=['number']).columns)
        categorical_count = len(dataframe.select_dtypes(include=['object', 'category']).columns)
        
        insights.append({
            "type": "general",
            "title": f"Mix de données: {numeric_count} numériques, {categorical_count} catégorielles",
            "description": f"Le dataset combine {numeric_count} variables numériques et {categorical_count} variables catégorielles.",
            "severity": "low",
            "recommendation": "Adapter les analyses selon les types de variables (corrélations pour numériques, distributions pour catégorielles)"
        })
        
        return insights
    
    async def generate_business_insights(
        self,
        dataframe: pd.DataFrame,
        business_context: str = None
    ) -> List[Dict[str, str]]:
        """Génère des insights orientés business"""
        try:
            context_prompt = f"""
            Contexte business: {business_context}
            """ if business_context else ""
            
            prompt = f"""
            {context_prompt}
            
            Analyse ce dataset d'un point de vue business et génère des insights actionnables.
            
            Dataset info:
            - {len(dataframe)} lignes, {len(dataframe.columns)} colonnes
            - Colonnes: {list(dataframe.columns)}
            - Types de données: {dict(dataframe.dtypes)}
            
            Échantillon:
            {dataframe.head(3).to_string()}
            
            Génère des insights business au format JSON avec:
            - Opportunités identifiées
            - Risques détectés  
            - Recommandations d'actions
            - KPIs suggérés
            - Segments de marché potentiels
            
            Format de réponse identique au précédent (JSON array d'objets).
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            insights_text = response.content.strip()
            
            import json
            try:
                return json.loads(insights_text)
            except json.JSONDecodeError:
                return self._parse_insights_fallback(insights_text)
                
        except Exception as e:
            logger.error(f"Erreur lors des insights business: {str(e)}")
            return [{
                "type": "business",
                "title": "Analyse business indisponible",
                "description": "Impossible de générer les insights business automatiquement",
                "severity": "low",
                "recommendation": "Effectuer une analyse manuelle des données"
            }]
"""
Service de création de graphiques
Réutilise la logique du nlq.py pour créer des visualisations intelligentes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from typing import Dict, List, Any, Optional
import logging

from ..models.api_models import ChartResponse

logger = logging.getLogger(__name__)

class ChartService:
    """Service pour la création de graphiques intelligents"""
    
    def __init__(self):
        # Configuration matplotlib pour l'API
        plt.style.use('default')
        
    async def create_chart_from_ai_recommendation(
        self,
        dataframe: pd.DataFrame,
        query: str,
        chart_type: Optional[str] = None,
        columns: Optional[List[str]] = None,
        recommendation: Optional[str] = None
    ) -> ChartResponse:
        """
        Crée un graphique basé sur les recommandations IA
        Réutilise la logique create_chart_from_ai_recommendation du nlq.py
        """
        try:
            # Si pas de recommandation fournie, utiliser les paramètres directs
            if recommendation:
                chart_type, columns, title = self._parse_recommendation(recommendation)
            else:
                title = f"Graphique pour: {query}"
            
            if not chart_type or not columns:
                return ChartResponse(
                    session_id="default",
                    chart_type="none",
                    chart_data={},
                    chart_config={},
                    description="Aucun graphique approprié trouvé",
                    columns_used=[],
                    success=False
                )
            
            # Nettoyer les noms de colonnes et vérifier qu'elles existent
            valid_columns = self._validate_columns(dataframe, columns)
            
            if not valid_columns:
                return ChartResponse(
                    session_id="default",
                    chart_type="none",
                    chart_data={},
                    chart_config={},
                    description="Colonnes spécifiées introuvables",
                    columns_used=[],
                    success=False
                )
            
            # Créer le graphique
            chart_data, chart_config = await self._create_chart(
                dataframe, chart_type, valid_columns, title
            )
            
            return ChartResponse(
                session_id="default",
                chart_type=chart_type,
                chart_data=chart_data,
                chart_config=chart_config,
                description=title,
                columns_used=valid_columns
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique: {str(e)}")
            return ChartResponse(
                session_id="default",
                chart_type="error",
                chart_data={},
                chart_config={},
                description=f"Erreur: {str(e)}",
                columns_used=[],
                success=False
            )
    
    def _parse_recommendation(self, recommendation: str) -> tuple:
        """Parse la recommandation IA pour extraire type, colonnes et titre"""
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
        
        return chart_type, columns, title
    
    def _validate_columns(self, dataframe: pd.DataFrame, columns: List[str]) -> List[str]:
        """Valide et nettoie les noms de colonnes"""
        valid_columns = []
        for col in columns:
            # Essayer de trouver la colonne même avec des variations de nom
            col_clean = col.strip('"\' ')
            if col_clean in dataframe.columns:
                valid_columns.append(col_clean)
            else:
                # Chercher une correspondance approximative
                for df_col in dataframe.columns:
                    if col_clean.lower() in df_col.lower() or df_col.lower() in col_clean.lower():
                        valid_columns.append(df_col)
                        break
        
        return valid_columns
    
    async def _create_chart(
        self, 
        df: pd.DataFrame, 
        chart_type: str, 
        columns: List[str], 
        title: str
    ) -> tuple:
        """
        Crée le graphique selon le type spécifié
        Logique extraite du nlq.py
        """
        # Créer le graphique avec un style cohérent
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('white')
        
        plot_created = False
        chart_data = {}
        
        try:
            if chart_type == 'bar' and len(columns) >= 1:
                plot_created = self._create_bar_chart(ax, df, columns)
                
            elif chart_type == 'boxplot' and len(columns) >= 2:
                plot_created = self._create_boxplot_chart(ax, df, columns)
                
            elif chart_type == 'violin' and len(columns) >= 2:
                plot_created = self._create_violin_chart(ax, df, columns)
                
            elif chart_type == 'pie' and len(columns) >= 1:
                plot_created = self._create_pie_chart(ax, df, columns, title)
                
            elif chart_type == 'area' and len(columns) >= 1:
                plot_created = self._create_area_chart(ax, df, columns)
                
            elif chart_type == 'density' and len(columns) >= 1:
                plot_created = self._create_density_chart(ax, df, columns)
                
            elif chart_type == 'line' and len(columns) >= 1:
                plot_created = self._create_line_chart(ax, df, columns)
                
            elif chart_type == 'scatter' and len(columns) >= 2:
                plot_created = self._create_scatter_chart(ax, df, columns)
                
            elif chart_type == 'hist' and len(columns) >= 1:
                plot_created = self._create_hist_chart(ax, df, columns)
                
            elif chart_type == 'heatmap' and len(columns) >= 2:
                plot_created = self._create_heatmap_chart(ax, df, columns)
            
            if plot_created:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Convertir en base64 pour l'API
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                chart_data = {
                    "image": image_base64,
                    "format": "png",
                    "width": 1200,
                    "height": 700
                }
                
            plt.close(fig)  # Libérer la mémoire
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique {chart_type}: {str(e)}")
            plt.close(fig)
        
        chart_config = {
            "type": chart_type,
            "columns": columns,
            "title": title,
            "created": plot_created
        }
        
        return chart_data, chart_config
    
    def _create_bar_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un graphique en barres"""
        col1 = columns[0]
        
        if len(columns) == 1:
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
                return True
            else:
                # Histogramme pour données numériques
                data_clean = df[col1].dropna()
                if len(data_clean) > 0:
                    ax.hist(data_clean, bins=min(30, len(data_clean.unique())), 
                           alpha=0.7, color='lightblue', edgecolor='navy')
                    ax.set_xlabel(col1)
                    ax.set_ylabel('Fréquence')
                    return True
        else:
            # Graphique groupé pour deux colonnes
            col2 = columns[1]
            if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
                grouped = df.groupby(col1)[col2].mean().sort_values(ascending=False).head(15)
                bars = ax.bar(range(len(grouped)), grouped.values, color='lightcoral', alpha=0.7)
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                ax.set_ylabel(f'Moyenne de {col2}')
                ax.set_xlabel(col1)
                return True
        
        return False
    
    def _create_boxplot_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un boxplot"""
        col1, col2 = columns[0], columns[1]
        if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
            groups = [group[col2].dropna() for name, group in df.groupby(col1)]
            labels = [name for name, group in df.groupby(col1)]
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            return True
        return False
    
    def _create_violin_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un violin plot"""
        col1, col2 = columns[0], columns[1]
        if df[col1].dtype in ['object', 'category'] and df[col2].dtype in ['int64', 'float64']:
            data_for_violin = [group[col2].dropna() for name, group in df.groupby(col1) if len(group) > 5]
            labels_for_violin = [name for name, group in df.groupby(col1) if len(group) > 5]
            if len(data_for_violin) > 0:
                ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)))
                ax.set_xticks(range(len(labels_for_violin)))
                ax.set_xticklabels(labels_for_violin, rotation=45, ha='right')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                return True
        return False
    
    def _create_pie_chart(self, ax, df: pd.DataFrame, columns: List[str], title: str) -> bool:
        """Crée un graphique en secteurs"""
        col1 = columns[0]
        if df[col1].dtype in ['object', 'category']:
            data = df[col1].value_counts().head(8)  # Max 8 segments
            if len(data) > 0:
                colors = plt.cm.Set3(range(len(data)))
                wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
                return True
        return False
    
    def _create_area_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un graphique en aires"""
        col1 = columns[0]
        if df[col1].dtype in ['int64', 'float64']:
            ax.fill_between(df.index, df[col1], alpha=0.7, color='lightblue')
            ax.plot(df.index, df[col1], color='navy', linewidth=2)
            ax.set_ylabel(col1)
            ax.set_xlabel('Index')
            return True
        return False
    
    def _create_density_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un graphique de densité"""
        col1 = columns[0]
        if df[col1].dtype in ['int64', 'float64']:
            data_clean = df[col1].dropna()
            if len(data_clean) > 10:
                data_clean.plot.density(ax=ax, color='purple', linewidth=2)
                ax.set_xlabel(col1)
                ax.set_ylabel('Densité')
                ax.fill_between(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata(), alpha=0.3, color='purple')
                return True
        return False
    
    def _create_line_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un graphique linéaire"""
        col1 = columns[0]
        if len(columns) == 1:
            ax.plot(df.index, df[col1], marker='o', linewidth=2, markersize=4)
            ax.set_ylabel(col1)
            ax.set_xlabel('Index')
        else:
            col2 = columns[1]
            ax.plot(df[col1], df[col2], marker='o', linewidth=2, markersize=4)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
        return True
    
    def _create_scatter_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un nuage de points"""
        col1, col2 = columns[0], columns[1]
        if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
            ax.scatter(df[col1], df[col2], alpha=0.6, s=50, color='coral')
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            return True
        return False
    
    def _create_hist_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée un histogramme"""
        col1 = columns[0]
        if df[col1].dtype in ['int64', 'float64']:
            data_clean = df[col1].dropna()
            if len(data_clean) > 0:
                ax.hist(data_clean, bins=min(30, len(data_clean.unique())), 
                       alpha=0.7, color='lightgreen', edgecolor='darkgreen')
                ax.set_xlabel(col1)
                ax.set_ylabel('Fréquence')
                return True
        return False
    
    def _create_heatmap_chart(self, ax, df: pd.DataFrame, columns: List[str]) -> bool:
        """Crée une heatmap de corrélation"""
        # Sélectionner uniquement les colonnes numériques
        numeric_data = df[columns].select_dtypes(include=['number'])
        if len(numeric_data.columns) >= 2:
            correlation = numeric_data.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
            return True
        return False
"""
Modèles de données pour l'API DataTalk
Définition des structures de requête et de réponse
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Modèles de base
class BaseRequest(BaseModel):
    """Modèle de base pour toutes les requêtes"""
    session_id: str = Field(..., description="Identifiant unique de session")

class BaseResponse(BaseModel):
    """Modèle de base pour toutes les réponses"""
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Modèles pour les requêtes de chat
class QueryRequest(BaseRequest):
    """Requête pour traitement en langage naturel"""
    query: str = Field(..., description="Question ou requête en langage naturel")
    context: Optional[str] = Field(None, description="Contexte additionnel pour la requête")

class QueryResponse(BaseResponse):
    """Réponse pour traitement en langage naturel"""
    session_id: str
    query: str
    answer: str
    code_executed: Optional[str] = Field(None, description="Code Python exécuté")
    has_chart: bool = False
    chart_data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

# Modèles pour les graphiques
class ChartRequest(BaseRequest):
    """Requête pour création de graphique"""
    query: str = Field(..., description="Description du graphique souhaité")
    chart_type: Optional[str] = Field(None, description="Type de graphique (bar, line, scatter, etc.)")
    columns: Optional[List[str]] = Field(None, description="Colonnes à utiliser")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Paramètres additionnels")

class ChartResponse(BaseResponse):
    """Réponse pour création de graphique"""
    session_id: str
    chart_type: str
    chart_data: Dict[str, Any]
    chart_config: Dict[str, Any]
    description: str
    columns_used: List[str]

# Modèles pour les insights
class InsightsRequest(BaseRequest):
    """Requête pour génération d'insights"""
    analysis_type: Optional[str] = Field("comprehensive", description="Type d'analyse (comprehensive, statistical, patterns)")
    focus_columns: Optional[List[str]] = Field(None, description="Colonnes à analyser spécifiquement")

class InsightsResponse(BaseResponse):
    """Réponse pour génération d'insights"""
    session_id: str
    insights: List[Dict[str, Any]]
    analysis_type: str
    summary: str
    recommendations: List[str]

# Modèles pour les questions suggérées
class QuestionsRequest(BaseRequest):
    """Requête pour génération de questions"""
    context: Optional[str] = Field(None, description="Contexte pour la génération")
    num_questions: Optional[int] = Field(5, description="Nombre de questions à générer")
    categories: Optional[List[str]] = Field(None, description="Catégories de questions")

class QuestionsResponse(BaseResponse):
    """Réponse pour génération de questions"""
    session_id: str
    questions: List[Dict[str, str]]
    categories: List[str]

# Modèles pour l'upload de données
class DataUploadResponse(BaseResponse):
    """Réponse pour upload de données"""
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    preview: List[Dict[str, Any]]
    automatic_insights: List[Dict[str, Any]]
    file_size: Optional[int] = None

# Modèles pour l'historique
class ChatMessage(BaseModel):
    """Message de chat"""
    role: str = Field(..., description="user ou assistant")
    content: str = Field(..., description="Contenu du message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = Field(None, description="Métadonnées additionnelles")

class SessionInfo(BaseModel):
    """Informations de session"""
    session_id: str
    dataset_shape: Optional[tuple] = None
    columns: Optional[List[str]] = None
    chat_messages: int = 0
    last_activity: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# Modèles d'erreur
class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée"""
    success: bool = False
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Modèles pour l'authentification
class TokenRequest(BaseModel):
    """Requête de token d'authentification"""
    api_key: str = Field(..., description="Clé API")
    user_id: Optional[str] = Field(None, description="Identifiant utilisateur")

class TokenResponse(BaseResponse):
    """Réponse avec token d'authentification"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: Optional[str] = None

# Modèles pour les statistiques
class UsageStats(BaseModel):
    """Statistiques d'utilisation"""
    total_queries: int
    total_sessions: int
    active_sessions: int
    total_uploads: int
    avg_response_time: float
    most_used_features: List[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Configuration et paramètres
class APIConfig(BaseModel):
    """Configuration de l'API"""
    max_file_size: int = Field(100 * 1024 * 1024, description="Taille max des fichiers (bytes)")
    max_rows: int = Field(1000000, description="Nombre max de lignes")
    allowed_file_types: List[str] = Field(["csv", "xlsx", "json"], description="Types de fichiers autorisés")
    session_timeout: int = Field(3600, description="Timeout de session (secondes)")
    rate_limit: int = Field(100, description="Limite de requêtes par minute")
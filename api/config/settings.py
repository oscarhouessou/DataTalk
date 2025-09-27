"""
Configuration de l'API DataTalk
Paramètres et variables d'environnement
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Settings(BaseSettings):
    """Configuration de l'application"""
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Permettre les champs supplémentaires
    
    # API Configuration
    app_name: str = "DataTalk API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Sécurité
    api_key: str = os.getenv("API_KEY", "")
    secret_key: str = os.getenv("SECRET_KEY", "datatalk-secret-key-change-in-production")
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    
    # Limites de l'API
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_rows: int = 1000000  # 1 million de lignes
    max_columns: int = 1000
    session_timeout: int = 3600  # 1 heure
    rate_limit_per_minute: int = 100
    
    # Types de fichiers autorisés
    allowed_file_extensions: list = ["csv", "xlsx", "json"]
    allowed_mime_types: list = [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json"
    ]
    
    # Base de données (pour extension future)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///datatalk.db")
    
    # Redis (pour le cache et les sessions)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "datatalk_api.log"
    
    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:8501", "*"]
    
    # Environnement
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Chemins
    temp_directory: str = "/tmp/datatalk"
    charts_directory: str = "/tmp/datatalk/charts"

@lru_cache()
def get_settings() -> Settings:
    """Récupère la configuration (avec cache)"""
    return Settings()

# Configuration par environnement
def get_database_config(environment: str = None):
    """Configuration de base de données par environnement"""
    settings = get_settings()
    env = environment or settings.environment
    
    if env == "production":
        return {
            "database_url": settings.database_url,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600
        }
    elif env == "testing":
        return {
            "database_url": "sqlite:///:memory:",
            "pool_size": 1,
            "max_overflow": 0
        }
    else:  # development
        return {
            "database_url": "sqlite:///datatalk_dev.db",
            "pool_size": 5,
            "max_overflow": 10
        }

def get_logging_config(environment: str = None):
    """Configuration du logging par environnement"""
    settings = get_settings()
    env = environment or settings.environment
    
    base_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": settings.log_file,
                "formatter": "detailed",
                "level": "INFO"
            }
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"]
        }
    }
    
    if env == "production":
        # En production, logging plus détaillé vers fichiers
        base_config["handlers"]["file"]["level"] = "WARNING"
        base_config["handlers"]["console"]["level"] = "ERROR"
    
    return base_config

# Configuration API par défaut
DEFAULT_API_CONFIG = {
    "title": "DataTalk API",
    "description": "API de traitement de données en langage naturel avec IA avancée",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json"
}
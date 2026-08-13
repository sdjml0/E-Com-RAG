import os
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Service Information
    PROJECT_NAME: str = "Multimodal E-Commerce RAG Microservice"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    
    # Qdrant Vector Database
    QDRANT_URL: str = os.getenv("QDRANT_URL", "./qdrant_db")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME: str = "ecommerce_products_v1"
    
    # Vector Dimensions
    TEXT_VECTOR_SIZE: int = 384  # Standard dense text embedding size (or 1536d)
    IMAGE_VECTOR_SIZE: int = 512  # Standard dense image embedding size (or 768d)
    
    # RRF Scoring Weights
    WEIGHT_TEXT: float = 0.45
    WEIGHT_IMAGE: float = 0.35
    WEIGHT_BM25: float = 0.20
    RRF_K: float = 60.0
    
    # Price Elasticity Penalty Configuration
    PRICE_PENALTY_ALPHA: float = 0.30
    MIN_PRICE_PENALTY_FACTOR: float = 0.50
    
    # Multimodal LLM (Gemini / Fallback)
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-3.6-flash")

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

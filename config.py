"""Configuration management for ML service."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://localhost:5432/inventory"

    # Service
    port: int = 8000  # Railway sets PORT env var
    ml_service_host: str = "0.0.0.0"

    # Model
    model_cache_dir: str = "./models/cache"
    enable_gpu: bool = False

    # Logging
    log_level: str = "INFO"

    # Training
    min_training_samples: int = 100
    test_size: float = 0.2
    random_state: int = 42

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

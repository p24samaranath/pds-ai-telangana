from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "PDS Optimization System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./pds_optimization.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL_FORECAST: int = 3600       # 1 hour
    REDIS_TTL_FRAUD: int = 1800          # 30 minutes
    REDIS_TTL_ROUTES: int = 21600        # 6 hours

    # Anthropic / Claude
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # ML Model Settings
    FRAUD_ALERT_THRESHOLD: float = 0.6
    FORECAST_LOOKAHEAD_DAYS: int = 90
    MODEL_RETRAIN_THRESHOLD_MAPE: float = 0.15  # Retrain if MAPE > 15%

    # Geospatial
    MAX_ACCEPTABLE_DISTANCE_KM: float = 5.0
    DEFAULT_MAP_CENTER_LAT: float = 17.3850   # Hyderabad, Telangana
    DEFAULT_MAP_CENTER_LON: float = 78.4867

    # Alert Settings
    FRAUD_SEVERITY_CRITICAL: float = 0.85
    FRAUD_SEVERITY_HIGH: float = 0.70
    FRAUD_SEVERITY_MEDIUM: float = 0.50
    FRAUD_SEVERITY_LOW: float = 0.30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

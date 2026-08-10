from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Dynamic Agentic System"
    app_version: str = "0.1.0"
    environment: str = "development"
    database_url: str


    model_config = SettingsConfigDict(
        env_file= BASE_DIR / ".env",
        extra="ignore",
    )


settings = Settings()
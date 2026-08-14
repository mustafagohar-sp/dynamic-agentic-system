from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Dynamic Agentic System"
    app_version: str = "0.1.0"
    environment: str = "development"
    database_url: str
    pinecone_api_key: str
    pinecone_index_name: str = "dynamic-agentic-system"

    openrouter_api_key : str
    openrouter_model : str = "openai/gpt-4o-mini"
    openrouter_base_url : str = "https://openrouter.ai/api/v1"

    model_config = SettingsConfigDict(
        env_file= BASE_DIR / ".env",
        extra="ignore",
    )


settings = Settings()
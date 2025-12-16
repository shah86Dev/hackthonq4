from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://user:password@localhost/dbname"

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    # Auth settings
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Application settings
    app_name: str = "Physical AI Textbook API"
    debug: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://user:password@localhost/book_rag"

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # OpenAI settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    # Application settings
    debug: bool = False
    max_tokens: int = 2048
    embedding_model: str = "text-embedding-ada-002"

    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_workers: int = 1

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # in seconds (1 hour)

    # Security
    secret_key: str = "your-secret-key-here"  # Should be set in production
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Book processing
    chunk_size: int = 500
    chunk_overlap: float = 0.1
    max_book_size: int = 1000000  # 1 million characters
    top_k_chunks: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
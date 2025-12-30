from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys and connection strings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    neon_postgres_url: str = os.getenv("NEON_POSTGRES_URL", "")

    # Application settings
    app_name: str = "Book-Embedded RAG Chatbot"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))
    server_workers: int = int(os.getenv("SERVER_WORKERS", "1"))

    # Qdrant settings
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embedding_size: int = 1536  # Size of OpenAI ada-002 embeddings

    # Book processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))  # Within the 500-1000 range specified
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))  # 200 character overlap as specified
    max_book_size: int = int(os.getenv("MAX_BOOK_SIZE", "1000000"))  # 1 million characters
    top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", "5"))  # Top 5 chunks for retrieval

    # Generation settings
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")  # Must be set in environment variables
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # CORS settings
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:3004",  # Docusaurus default
        "http://127.0.0.1:3004",
        "http://localhost:3001",  # Additional common port
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3002",  # Additional common port
        "http://127.0.0.1:3002",
        "*"  # Allow all origins for development - should be restricted in production
    ]  # Default for development

    # Ray settings for distributed processing
    ray_address: str = os.getenv("RAY_ADDRESS", "auto")  # Use "auto" for local cluster

    model_config = {"env_file": ".env", "extra": "allow"}


settings = Settings()
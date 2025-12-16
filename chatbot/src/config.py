import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Configuration settings for the chatbot and subagents"""

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Qdrant settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    TEXTBOOK_COLLECTION: str = os.getenv("TEXTBOOK_COLLECTION", "textbook_content")

    # Application settings
    APP_NAME: str = "Physical AI Textbook RAG Chatbot"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Content settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", 3000))  # tokens
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", 500))

    # Grounding settings
    GROUNDING_THRESHOLD: float = float(os.getenv("GROUNDING_THRESHOLD", 0.1))

    # Subagent settings
    SUBAGENT_TIMEOUT: int = int(os.getenv("SUBAGENT_TIMEOUT", 300))  # seconds
    MAX_SUBAGENT_RETRIES: int = int(os.getenv("MAX_SUBAGENT_RETRIES", 3))

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    def validate(self) -> None:
        """Validate configuration settings"""
        errors = []

        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")

        if not self.OPENAI_MODEL:
            errors.append("OPENAI_MODEL must be specified")

        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

# Create a global settings instance
settings = Settings()

# Validate settings on import
settings.validate()
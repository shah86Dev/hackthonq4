"""
Book-Embedded RAG Chatbot - Main Application

This module defines the FastAPI application with all necessary endpoints
and configuration for the RAG chatbot system.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import os
from typing import Optional

from src.database import get_db
from src.config.settings import settings
from src.api.endpoints.chat import router as chat_router
from src.api.endpoints.ingest import router as ingest_router
from src.api.endpoints.health import router as health_router
from src.api.middleware import add_rate_limiting_middleware

# Create FastAPI app
app = FastAPI(
    title="Book-Embedded RAG Chatbot API",
    description="API for Book-Embedded RAG Chatbot with Qdrant vector store and OpenAI integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add rate limiting middleware
add_rate_limiting_middleware(app)

# Include API routes
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(health_router, prefix="/api/v1", tags=["health"])

@app.get("/")
def read_root():
    """
    Root endpoint that returns basic information about the API
    """
    return {
        "message": "Book-Embedded RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "ingest": "/api/v1/ingest"
        }
    }

@app.get("/api/v1/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Book-Embedded RAG Chatbot Backend",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

@app.get("/api/v1/config")
def get_config():
    """
    Configuration endpoint to check API settings
    """
    return {
        "debug": settings.debug,
        "database_url": "SET" if settings.database_url else "NOT SET",
        "openai_api_key": "SET" if settings.openai_api_key else "NOT SET",
        "qdrant_host": settings.qdrant_host,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "cors_origins": settings.cors_origins
    }

@app.on_event("startup")
async def startup_event():
    """
    Startup event to initialize services
    """
    print("Starting up Book-Embedded RAG Chatbot API...")

    # Initialize Qdrant collection if needed
    try:
        from src.services.qdrant_client import qdrant_service
        print(f"Connected to Qdrant: {qdrant_service.collection_name}")
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant: {e}")

    # Initialize database if needed
    try:
        from src.database import engine
        from src.models import Base
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        print("Database connection established")
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event
    """
    print("Shutting down Book-Embedded RAG Chatbot API...")

if __name__ == "__main__":
    # Configuration for local development
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
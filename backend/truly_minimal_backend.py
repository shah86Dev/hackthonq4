"""
Truly Minimal Book-Embedded RAG Chatbot Backend for local development
This version has no external dependencies to avoid all compatibility issues
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with minimal configuration
app = FastAPI(
    title="Book-Embedded RAG Chatbot API - Truly Minimal Version",
    description="API for Book-Embedded RAG Chatbot - Fallback version with no external dependencies",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3004",  # Docusaurus default
        "http://127.0.0.1:3004",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"]
)

# Add exception handler to catch errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal Server Error: {str(exc)}"}
    )

# Simple endpoints
@app.get("/")
def read_root():
    """
    Root endpoint that returns basic information about the API
    """
    return {
        "message": "Book-Embedded RAG Chatbot API - Truly Minimal Version",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/health",
            "chat": "/api/v1/chat",
            "ingest": "/api/v1/ingest"
        }
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Book-Embedded RAG Chatbot Backend - Truly Minimal",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

# Simple fallback endpoints
@app.post("/api/v1/chat")
def chat_endpoint():
    """
    Fallback chat endpoint
    """
    return {
        "response": "Chat service is not currently available",
        "status": "fallback",
        "message": "This is a fallback endpoint. The full service requires proper AI and database configuration."
    }

@app.post("/api/v1/ingest")
def ingest_endpoint():
    """
    Fallback ingest endpoint
    """
    return {
        "status": "not_available",
        "message": "Ingest service is not currently available in this fallback version"
    }

if __name__ == "__main__":
    # Configuration for local development
    host = "0.0.0.0"
    port = 8000
    reload = True

    uvicorn.run(
        "truly_minimal_backend:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
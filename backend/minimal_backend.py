"""
Minimal Book-Embedded RAG Chatbot Backend for local development
This version excludes the auth module to avoid compatibility issues
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Book-Embedded RAG Chatbot API - Minimal Version",
    description="API for Book-Embedded RAG Chatbot with Qdrant vector store and OpenAI integration (minimal version)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware with proper configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Create React App default
        "http://localhost:3004", # Docusaurus default
        "*"
    ],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Allow all headers including authorization
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

# Simple health check
@app.get("/")
def read_root():
    """
    Root endpoint that returns basic information about the API
    """
    return {
        "message": "Book-Embedded RAG Chatbot API - Minimal Version",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "ingest": "/api/v1/ingest"
        }
    }

@app.get("/health")
def health_check():
    """
    Simple health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Book-Embedded RAG Chatbot Backend - Minimal",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

# Try to import endpoints with error handling
try:
    from src.api.endpoints.chat import router as chat_router
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    logger.info("Chat router loaded successfully")
except Exception as e:
    logger.error(f"Error loading chat router: {e}")
    # Create a simple fallback chat endpoint
    @app.post("/api/v1/chat")
    def chat_fallback():
        return {"response": "Chat endpoint not available", "error": str(e)}

try:
    from src.api.endpoints.ingest import router as ingest_router
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingestion"])
    logger.info("Ingest router loaded successfully")
except Exception as e:
    logger.error(f"Error loading ingest router: {e}")
    # Create a simple fallback ingest endpoint
    @app.post("/api/v1/ingest")
    def ingest_fallback():
        return {"status": "Ingest endpoint not available", "error": str(e)}

try:
    from src.api.endpoints.health import router as health_router
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    logger.info("Health router loaded successfully")
except Exception as e:
    logger.error(f"Error loading health router: {e}")
    # Health endpoint already defined above

if __name__ == "__main__":
    # Configuration for local development
    host = "0.0.0.0"
    port = 8000
    reload = True

    uvicorn.run(
        "minimal_backend:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
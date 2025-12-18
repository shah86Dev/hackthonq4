from fastapi import FastAPI
from .endpoints import ingest, query, health
from .middleware import add_middleware
from .error_handlers import add_exception_handlers
from src.config.settings import settings


app = FastAPI(
    title="Book-Embedded RAG Chatbot API",
    description="API for embedding RAG chatbot in digital books",
    version="1.0.0"
)

# Add middleware
app = add_middleware(app)

# Add exception handlers
app = add_exception_handlers(app)

# Include API routers
app.include_router(ingest.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
def read_root():
    return {"message": "Book-Embedded RAG Chatbot API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.debug
    )
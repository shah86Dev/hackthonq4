from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from . import crud, models, schemas
from .database import get_db
from .config import settings
import uvicorn


# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="API for Physical AI & Humanoid Robotics Textbook",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # Configure allowed origins in environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specify allowed methods
    allow_headers=["*"],
)

# Include API routes
from .api import auth, chapters, content, translation, personalization

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chapters.router, prefix="/api/chapters", tags=["chapters"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(translation.router, prefix="/api/translation", tags=["translation"])
app.include_router(personalization.router, prefix="/api/personalization", tags=["personalization"])


@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook API", "version": "1.0.0"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Physical AI Textbook Backend"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main_no_db:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.debug else False
    )
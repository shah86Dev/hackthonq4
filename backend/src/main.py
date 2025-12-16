from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import engine, get_db
from .config import settings
import uvicorn


# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API for Physical AI & Humanoid Robotics Textbook",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.debug else False
    )
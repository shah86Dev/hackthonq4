from fastapi import FastAPI
import uvicorn


# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="API for Physical AI & Humanoid Robotics Textbook",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook API", "version": "1.0.0"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Physical AI Textbook Backend"}


if __name__ == "__main__":
    uvicorn.run(
        "minimal_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Simple Backend API")

@app.get("/")
def read_root():
    return {"message": "Simple Backend is running!", "status": "success"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Simple Backend"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Changed to port 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
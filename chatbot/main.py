import uvicorn
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the chatbot application"""

    # Set environment variables for Qdrant if not set
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")

    print(f"Starting Physical AI Textbook RAG Chatbot...")
    print(f"Qdrant connection: {qdrant_host}:{qdrant_port}")

    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8001,  # Using port 8001 to differentiate from other services
        reload=True,  # Set to False in production
        log_level="info"
    )

if __name__ == "__main__":
    main()
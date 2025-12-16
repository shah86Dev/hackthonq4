from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os

from .vector_store import VectorStore
from .agents import RAGAgent
from .config import settings


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "en"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    source_chunks: List[str]
    is_grounding_valid: bool


class ContentChunk(BaseModel):
    id: str
    content: str
    content_urdu: Optional[str] = None
    chapter_id: str
    chunk_type: str
    metadata: Dict[str, Any]


class ContentUploadRequest(BaseModel):
    chunks: List[ContentChunk]
    language: str = "en"


app = FastAPI(
    title="Physical AI Textbook RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for Physical AI & Humanoid Robotics textbook",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store and agent
vector_store = VectorStore(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT
)
rag_agent = RAGAgent(vector_store)


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store and load embeddings"""
    await vector_store.initialize()
    print("Chatbot system initialized successfully")


@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook RAG Chatbot", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAG Chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user queries and returns AI-generated responses
    grounded in textbook content only.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process the query using the RAG agent
        response = await rag_agent.process_query(
            query=request.message,
            session_id=session_id,
            language=request.language
        )

        return ChatResponse(
            response=response.response_text,
            session_id=session_id,
            source_chunks=response.source_chunk_ids,
            is_grounding_valid=response.is_grounding_valid
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-content")
async def upload_content(request: ContentUploadRequest):
    """
    Upload textbook content to the vector store for RAG retrieval
    """
    try:
        # Prepare content for embedding
        content_to_store = []
        for chunk in request.chunks:
            content_to_store.append({
                "id": chunk.id,
                "content": chunk.content if request.language == "en" else (chunk.content_urdu or chunk.content),
                "chapter_id": chunk.chapter_id,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata
            })

        # Add content to vector store
        await vector_store.add_content(content_to_store)

        return {"status": "success", "count": len(content_to_store)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """
    Retrieve chat history for a specific session
    """
    try:
        history = await rag_agent.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session
    """
    try:
        await rag_agent.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
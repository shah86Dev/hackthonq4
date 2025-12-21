from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from typing import Optional
from src.config.database import get_db
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService
from pydantic import BaseModel


router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None
    language: Optional[str] = "en"
    book_id: Optional[str] = "12345678-1234-5678-1234-567812345678"  # Default placeholder book ID


class ChatResponse(BaseModel):
    response: str
    source_chunks: list = []
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint for the book RAG chatbot
    This endpoint matches the frontend expectations and implements the spec requirements.
    """
    try:
        # Initialize services
        retrieval_service = RetrievalService()
        generation_service = GenerationService()
        citation_service = CitationService()

        # Validate the book_id is a valid UUID
        try:
            book_id = uuid.UUID(request.book_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

        # Determine the mode based on whether selected_text is provided
        if request.selected_text:
            # Use selected text mode
            retrieved_chunks = retrieval_service.retrieve_chunks_selected_text(
                selected_text=request.selected_text,
                query=request.question
            )
            mode = "selected-text"
        else:
            # Use full-book RAG mode
            retrieved_chunks = retrieval_service.retrieve_chunks_full_book(
                book_id=book_id,
                query=request.question,
                top_k=5
            )
            mode = "full-book"

        # If no chunks were retrieved (and we're not in selected-text mode), return an error
        if not retrieved_chunks and mode != "selected-text":
            answer_text = "I cannot find this information in the book."
        else:
            # Generate answer using the retrieved chunks
            generation_result = generation_service.generate_answer(
                question=request.question,
                context_chunks=retrieved_chunks,
                mode=mode
            )
            answer_text = generation_result["answer_text"]

        # Format citations using the citation service
        citations = citation_service.format_citations(retrieved_chunks)
        source_chunks = citations

        # Create response
        response = ChatResponse(
            response=answer_text,
            source_chunks=source_chunks,
            session_id=request.session_id or str(uuid.uuid4())
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
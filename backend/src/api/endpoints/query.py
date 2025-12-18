from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from typing import List, Optional
from src.config.database import get_db
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService
from pydantic import BaseModel


router = APIRouter()


class QueryRequest(BaseModel):
    book_id: str
    question: str
    mode: str = "full-book"  # 'full-book' or 'selected-text'
    selected_text: Optional[str] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    status: str
    answer: str
    citations: List[dict]
    confidence_score: float
    retrieved_chunks: List[dict]
    query_id: str


@router.post("/query", response_model=QueryResponse)
async def query_book(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Query the book content using either full-book RAG or selected-text RAG
    """
    if request.mode not in ["full-book", "selected-text"]:
        raise HTTPException(status_code=400, detail="Mode must be 'full-book' or 'selected-text'")

    if request.mode == "selected-text" and not request.selected_text:
        raise HTTPException(status_code=400, detail="selected_text is required for selected-text mode")

    try:
        # Validate book_id is a valid UUID
        book_uuid = uuid.UUID(request.book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

    try:
        # Initialize services
        retrieval_service = RetrievalService()
        generation_service = GenerationService()
        citation_service = CitationService()

        # Retrieve relevant chunks based on mode
        if request.mode == "full-book":
            retrieved_chunks = retrieval_service.retrieve_chunks_full_book(
                book_id=book_uuid,
                query=request.question,
                top_k=5
            )
        else:  # selected-text mode
            retrieved_chunks = retrieval_service.retrieve_chunks_selected_text(
                selected_text=request.selected_text,
                query=request.question
            )

        # If no chunks were retrieved (and we're not in selected-text mode), return an error
        if not retrieved_chunks and request.mode != "selected-text":
            raise HTTPException(
                status_code=422,
                detail="No relevant content found in book. I cannot find this information in the book."
            )

        # Generate answer using the retrieved chunks
        generation_result = generation_service.generate_answer(
            question=request.question,
            context_chunks=retrieved_chunks,
            mode=request.mode
        )

        # Format citations
        citations = citation_service.extract_citations_from_answer(
            answer=generation_result["answer_text"],
            retrieved_chunks=retrieved_chunks
        )

        # Create response
        response = QueryResponse(
            status="success",
            answer=generation_result["answer_text"],
            citations=citations,
            confidence_score=generation_result["confidence_score"],
            retrieved_chunks=retrieved_chunks,
            query_id=str(uuid.uuid4())  # Generate a unique query ID
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
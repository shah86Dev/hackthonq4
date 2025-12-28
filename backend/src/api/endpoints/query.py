from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from typing import List, Optional
from datetime import datetime
from src.database import get_db
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService
from src.models.query_log import QueryLog
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.config.settings import settings


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


# Create a limiter specifically for query endpoint with default IP-based rate limiting
query_limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_requests}/hour"])

# For per-session rate limiting, we'll use the middleware approach which can access both IP and session data
@router.post("/query", response_model=QueryResponse)
@query_limiter.limit(f"{settings.rate_limit_requests}/hour")  # Use configurable rate limit
async def query_book(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Query the book content using either full-book RAG or selected-text RAG
    """
    start_time = datetime.utcnow()

    if request.mode not in ["full-book", "selected-text"]:
        raise HTTPException(status_code=400, detail="Mode must be 'full-book' or 'selected-text'")

    if request.mode == "selected-text" and not request.selected_text:
        raise HTTPException(status_code=400, detail="selected_text is required for selected-text mode")

    try:
        # Validate book_id is a valid UUID
        book_uuid = uuid.UUID(request.book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

    # Validate that the book exists in the database
    from src import crud
    if not crud.book_exists(db, str(book_uuid)):
        raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found.")

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
            # Log the failed query
            try:
                query_log = QueryLog(
                    book_id=book_uuid,
                    mode=request.mode,
                    question=request.question,
                    selected_text=request.selected_text,
                    retrieved_chunk_ids=None,
                    answer="No relevant content found in book. I cannot find this information in the book.",
                    latency=(datetime.utcnow() - start_time).total_seconds(),
                    tokens_used=0,
                    confidence_score=0.0,
                    user_id=request.user_id,
                    session_id=str(uuid.uuid4()),
                    is_successful=False
                )
                db.add(query_log)
                db.commit()
            except Exception as e:
                print(f"Failed to log failed query: {e}")
                db.rollback()

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
        query_id = str(uuid.uuid4())  # Generate a unique query ID
        response = QueryResponse(
            status="success",
            answer=generation_result["answer_text"],
            citations=citations,
            confidence_score=generation_result["confidence_score"],
            retrieved_chunks=retrieved_chunks,
            query_id=query_id
        )

        # Log the query for analytics
        try:
            # Extract chunk IDs from retrieved chunks
            chunk_ids = []
            for chunk in retrieved_chunks:
                if 'chunk_id' in chunk and chunk['chunk_id']:
                    # Extract UUID from chunk_id if it's in the format "book-uuid-index"
                    chunk_id_str = chunk['chunk_id']
                    if '-' in chunk_id_str:
                        # Try to extract the UUID part from the chunk_id
                        parts = chunk_id_str.split('-')
                        if len(parts) >= 2:
                            # If it's in the format "book-uuid-index", take the UUID part
                            try:
                                chunk_uuid = uuid.UUID(parts[0])
                                chunk_ids.append(chunk_uuid)
                            except ValueError:
                                # If not a valid UUID, try to parse differently
                                try:
                                    # Try to find a UUID pattern in the string
                                    import re
                                    uuid_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', chunk_id_str, re.IGNORECASE)
                                    if uuid_match:
                                        chunk_ids.append(uuid.UUID(uuid_match.group()))
                                    else:
                                        # Generate a placeholder UUID if we can't extract a real one
                                        chunk_ids.append(uuid.uuid4())
                                except:
                                    chunk_ids.append(uuid.uuid4())
                    else:
                        try:
                            chunk_ids.append(uuid.UUID(chunk_id_str))
                        except ValueError:
                            chunk_ids.append(uuid.uuid4())

            # Calculate actual latency
            latency = (datetime.utcnow() - start_time).total_seconds()

            # Create query log entry
            query_log = QueryLog(
                book_id=book_uuid,
                mode=request.mode,
                question=request.question,
                selected_text=request.selected_text,
                retrieved_chunk_ids=chunk_ids if chunk_ids else None,
                answer=generation_result["answer_text"],
                latency=latency,
                tokens_used=generation_result.get("tokens_used", 0),
                confidence_score=generation_result["confidence_score"],
                user_id=request.user_id,
                session_id=query_id,  # Use query_id as session_id if no specific session tracking
                is_successful=True
            )
            db.add(query_log)
            db.commit()
        except Exception as e:
            # If logging fails, don't fail the entire query
            print(f"Failed to log query: {e}")
            db.rollback()

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
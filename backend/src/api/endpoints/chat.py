from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from typing import Optional
from datetime import datetime
from src.database import get_db
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService
from src.models.query_log import QueryLog
from pydantic import BaseModel
from slowapi import limit
from src.config.settings import settings
from fastapi import Request
from src.api.middleware import limiter


router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None
    language: Optional[str] = "en"
    book_id: str  # Required book ID - no default provided


class ChatResponse(BaseModel):
    response: str
    source_chunks: list = []
    session_id: str


from src.config.settings import settings

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_requests}/hour")  # Use the imported limiter
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db), fastapi_request: Request = None):
    """
    Main chat endpoint for the book RAG chatbot
    This endpoint matches the frontend expectations and implements the spec requirements.
    """
    start_time = datetime.utcnow()

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

        # Validate that the book exists in the database
        from src import crud
        if not crud.book_exists(db, str(book_id)):
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found.")

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
        session_id = request.session_id or str(uuid.uuid4())
        response = ChatResponse(
            response=answer_text,
            source_chunks=source_chunks,
            session_id=session_id
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
                book_id=book_id,
                mode=mode,
                question=request.question,
                selected_text=request.selected_text,
                retrieved_chunk_ids=chunk_ids if chunk_ids else None,
                answer=answer_text,
                latency=latency,
                tokens_used=generation_result.get("tokens_used", 0) if 'generation_result' in locals() else 0,
                confidence_score=generation_result.get("confidence_score", 0.0) if 'generation_result' in locals() else 0.0,
                user_id=None,  # No user_id in chat request
                session_id=session_id,
                is_successful=answer_text != "I cannot find this information in the book."
            )
            db.add(query_log)
            db.commit()
        except Exception as e:
            # If logging fails, don't fail the entire query
            print(f"Failed to log chat query: {e}")
            db.rollback()

        return response

    except Exception as e:
        # Log the error query
        try:
            query_log = QueryLog(
                book_id=uuid.UUID(request.book_id) if request.book_id else None,
                mode="selected-text" if request.selected_text else "full-book",
                question=request.question,
                selected_text=request.selected_text,
                retrieved_chunk_ids=None,
                answer=f"Error: {str(e)}",
                latency=(datetime.utcnow() - start_time).total_seconds(),
                tokens_used=0,
                confidence_score=0.0,
                user_id=None,
                session_id=request.session_id or str(uuid.uuid4()),
                is_successful=False
            )
            db.add(query_log)
            db.commit()
        except:
            pass  # Don't let logging failure mask the original error

        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
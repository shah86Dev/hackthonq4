from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from uuid import UUID
import uuid
from typing import Optional
from datetime import datetime
from src.database import get_db
from src.services.ingestion_service import IngestionService
from src.models.book import Book
from pydantic import BaseModel
from src.config.settings import settings
from fastapi import Request
import logging
from src.api.middleware import limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestBookRequest(BaseModel):
    book_id: str
    title: str
    version: str
    content: str
    file_format: str = "text"  # pdf, md, txt, text
    metadata: dict = None


class IngestRequest(BaseModel):
    title: str
    author: Optional[str] = None
    file_format: str  # pdf, md, txt
    version: Optional[str] = "1.0"


class IngestResponse(BaseModel):
    status: str
    book_id: str
    chunks_processed: int
    file_format: str
    message: str




@router.post("/ingest/book")
@limiter.limit(f"{settings.rate_limit_requests}/hour")  # Apply rate limiting
async def ingest_book(request: IngestBookRequest, db: Session = Depends(get_db)):
    """
    Ingest a book into the RAG system with raw text content
    """
    try:
        # Validate the book_id is a valid UUID
        book_uuid = UUID(request.book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

    try:
        ingestion_service = IngestionService()
        result = ingestion_service.ingest_book(
            db=db,
            book_id=book_uuid,
            title=request.title,
            version=request.version,
            content=request.content,
            file_format=request.file_format,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit(f"{settings.rate_limit_requests}/hour")  # Apply rate limiting
async def ingest_endpoint(
    request: Request,  # For rate limiting
    file: UploadFile = File(...),
    title: str = "",
    author: str = "",
    file_format: str = "",
    version: str = "1.0",
    db: Session = Depends(get_db)
):
    """
    Ingest endpoint to upload and process books in various formats (PDF, Markdown, TXT)
    With performance monitoring and batch processing capabilities
    """
    start_time = datetime.utcnow()

    try:
        # Validate file format if not provided in query param, try to detect from filename
        if not file_format:
            if file.filename:
                if file.filename.lower().endswith('.pdf'):
                    file_format = 'pdf'
                elif file.filename.lower().endswith('.md') or file.filename.lower().endswith('.markdown'):
                    file_format = 'markdown'
                elif file.filename.lower().endswith('.txt'):
                    file_format = 'txt'
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file format. Supported formats: PDF, Markdown, TXT")
            else:
                raise HTTPException(status_code=400, detail="File format not specified and could not be detected from filename")

        # Validate file format
        if file_format.lower() not in ['pdf', 'markdown', 'md', 'txt']:
            raise HTTPException(status_code=400, detail="Unsupported file format. Supported formats: PDF, Markdown, TXT")

        # Normalize file format to standard values
        if file_format.lower() == 'md':
            file_format = 'markdown'

        # Read file content
        content = await file.read()

        # Validate file size (prevent extremely large files)
        if len(content) > settings.max_book_size:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {settings.max_book_size} characters.")

        # Generate a unique book ID
        book_id = uuid.uuid4()

        # Initialize the ingestion service
        ingestion_service = IngestionService()

        # Record processing start time
        processing_started = datetime.utcnow()

        # Process the content based on format
        processed_content = ingestion_service.process_content(content, file_format, file.filename)

        # Check if this is a large book that requires batch processing
        content_length = len(processed_content)
        is_large_book = content_length > 50000  # More than 50k characters

        logger.info(f"Processing {'large' if is_large_book else 'small'} book with {content_length} characters")

        # Ingest the book
        result = ingestion_service.ingest_book(
            db=db,
            book_id=book_id,
            title=title or file.filename or "Untitled Book",
            version=version,
            content=processed_content,
            file_format=file_format,
            metadata={
                "original_filename": file.filename,
                "file_size": len(content),
                "content_length": content_length,
                "processing_started": processing_started.isoformat(),
                "author": author,
                "is_large_book": is_large_book
            }
        )

        # Calculate processing time
        processing_time = (datetime.utcnow() - processing_started).total_seconds()

        # Update book record with processing completion time
        book_record = db.query(Book).filter(Book.id == book_id).first()
        if book_record:
            book_record.processing_completed_at = datetime.utcnow()
            book_record.total_chunks = result.get("chunks_processed", 0)
            book_record.total_tokens = content_length  # Approximate token count
            book_record.processing_status = "completed"
            book_record.processing_time_seconds = processing_time  # Add processing time to book record
            db.commit()

        # Log performance metrics
        total_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Successfully ingested book {book_id} with {result['chunks_processed']} chunks. "
                   f"Content length: {content_length}, Processing time: {processing_time}s, "
                   f"Total time: {total_time}s, Chunks per second: {result['chunks_processed']/processing_time if processing_time > 0 else 0:.2f}")

        return IngestResponse(
            status=result["status"],
            book_id=result["book_id"],
            chunks_processed=result["chunks_processed"],
            file_format=result["file_format"],
            message=f"{result['message']} Processing time: {processing_time:.2f}s for {content_length} characters."
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"Error in ingest endpoint after {processing_time}s: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed after {processing_time:.2f}s: {str(e)}")


@router.post("/ingest-text", response_model=IngestResponse)
@limiter.limit(f"{settings.rate_limit_requests}/hour")  # Apply rate limiting
async def ingest_text_endpoint(
    request: Request,  # For rate limiting
    ingest_request: IngestRequest,
    content: str,
    db: Session = Depends(get_db)
):
    """
    Alternative endpoint to ingest raw text content directly (for testing or specific use cases)
    """
    try:
        # Validate file format
        if ingest_request.file_format.lower() not in ['pdf', 'markdown', 'md', 'txt', 'text']:
            raise HTTPException(status_code=400, detail="Unsupported file format. Supported formats: PDF, Markdown, TXT, Text")

        # Normalize file format to standard values
        file_format = ingest_request.file_format.lower()
        if file_format == 'md':
            file_format = 'markdown'
        elif file_format == 'text':
            file_format = 'txt'

        # Generate a unique book ID
        book_id = uuid.uuid4()

        # Initialize the ingestion service
        ingestion_service = IngestionService()

        # Ingest the book with the provided text content
        result = ingestion_service.ingest_book(
            db=db,
            book_id=book_id,
            title=ingest_request.title,
            version=ingest_request.version or "1.0",
            content=content,
            file_format=file_format,
            metadata={
                "author": ingest_request.author,
                "source": "direct_text_input"
            }
        )

        # Update book record with processing completion time
        book_record = db.query(Book).filter(Book.id == book_id).first()
        if book_record:
            book_record.processing_completed_at = datetime.utcnow()
            book_record.total_chunks = result.get("chunks_processed", 0)
            book_record.processing_status = "completed"
            db.commit()

        logger.info(f"Successfully ingested text book {book_id} with {result['chunks_processed']} chunks")

        return IngestResponse(
            status=result["status"],
            book_id=result["book_id"],
            chunks_processed=result["chunks_processed"],
            file_format=result["file_format"],
            message=result["message"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in ingest text endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")
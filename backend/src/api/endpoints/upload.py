from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from src.config.database import get_db
from src.services.document_processor import DocumentProcessor
from src.services.ingestion_service import IngestionService
from pydantic import BaseModel
import tempfile
import os


router = APIRouter()
processor = DocumentProcessor()


class UploadBookRequest(BaseModel):
    title: str
    version: str
    book_id: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/upload/book")
async def upload_book(
    request: UploadBookRequest,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a book file (PDF/Markdown) for the RAG system
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.md', '.markdown']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Generate a unique book ID if not provided
        if not request.book_id:
            book_uuid = uuid.uuid4()
        else:
            try:
                book_uuid = uuid.UUID(request.book_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read the uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process the document
            full_text, chunks = processor.process_document(
                temp_file_path,
                chunk_size=750,  # Using 750 chars which is within the 500-1000 range specified
                overlap=200      # 200 char overlap as specified
            )

            # Use the ingestion service to store the book in the system
            ingestion_service = IngestionService()
            result = ingestion_service.ingest_book(
                db=db,
                book_id=book_uuid,
                title=request.title,
                version=request.version,
                content=full_text,
                metadata=request.metadata
            )

            return {
                "status": "success",
                "book_id": str(book_uuid),
                "title": request.title,
                "chunks_processed": len(chunks),
                "message": f"Successfully processed and ingested book '{request.title}' with {len(chunks)} chunks"
            }

        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
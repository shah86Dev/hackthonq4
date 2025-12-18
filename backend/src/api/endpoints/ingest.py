from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
import uuid
from src.config.database import get_db
from src.services.ingestion_service import IngestionService
from pydantic import BaseModel


router = APIRouter()


class IngestBookRequest(BaseModel):
    book_id: str
    title: str
    version: str
    content: str
    metadata: dict = None


@router.post("/ingest/book")
async def ingest_book(request: IngestBookRequest, db: Session = Depends(get_db)):
    """
    Ingest a book into the RAG system
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
            metadata=request.metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
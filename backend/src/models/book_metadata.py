from sqlalchemy import Column, String, DateTime, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.database import Base
import uuid
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class BookMetadata(Base):
    """
    SQLAlchemy model for storing book metadata and processing information
    """
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    author = Column(String(200))
    file_path = Column(String(1000))  # Path to original file
    file_type = Column(String(10), nullable=False)  # pdf, md, etc.
    total_pages = Column(Integer)
    total_chunks = Column(Integer, default=0)  # Number of chunks created
    total_tokens = Column(Integer, default=0)  # Total tokens in the book
    processing_status = Column(String(20), default='pending')  # pending, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BookMetadataPydantic(BaseModel):
    """
    Pydantic model for Book Metadata with validation rules
    """
    id: Optional[str] = None
    title: str
    author: Optional[str] = None
    file_path: Optional[str] = None
    file_type: str
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = 0
    total_tokens: Optional[int] = 0
    processing_status: str = 'pending'
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "book-123",
                "title": "The Complete Guide to AI",
                "author": "John Doe",
                "file_type": "pdf",
                "total_pages": 300,
                "total_chunks": 150,
                "total_tokens": 45000,
                "processing_status": "completed"
            }
        }


class BookMetadataCreate(BaseModel):
    """
    Model for creating a new book metadata entry
    """
    title: str
    author: Optional[str] = None
    file_type: str
    total_pages: Optional[int] = None
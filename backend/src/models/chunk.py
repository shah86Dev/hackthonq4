from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func
from src.database import Base
import uuid
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer)  # Page number in the original book
    section = Column(String(200))  # Section/chapter title
    text = Column(Text, nullable=False)  # The actual text content of the chunk
    embedding = Column(JSON)  # Store embedding as JSON array for Neon compatibility
    position = Column(Integer, nullable=False)  # Sequential position in the book
    metadata = Column(JSON)  # Additional metadata (word count, etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BookChunkPydantic(BaseModel):
    """
    Pydantic model for Book Chunk representing a segment of book content with embedding vector and metadata for retrieval.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    text_content: str = Field(..., min_length=500, max_length=1000, description="The actual text content of the chunk")
    book_id: str = Field(..., description="Reference to the parent book")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in the original book")
    section: Optional[str] = Field(None, description="Section/chapter title")
    position: int = Field(..., ge=1, description="Sequential position in the book")
    embedding_vector: List[float] = Field(..., description="Embedding vector for semantic search")
    metadata: Optional[dict] = Field(None, description="Additional metadata (word count, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk-123",
                "text_content": "This is a sample chunk of book content that contains important information...",
                "book_id": "book-456",
                "page_number": 25,
                "section": "Chapter 3: Introduction",
                "position": 1,
                "embedding_vector": [0.1, 0.2, 0.3],  # Simplified example
                "metadata": {"word_count": 150}
            }
        }


class BookChunkCreate(BaseModel):
    """
    Model for creating a new book chunk.
    """
    text_content: str = Field(..., min_length=500, max_length=1000, description="The actual text content of the chunk")
    book_id: str = Field(..., description="Reference to the parent book")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in the original book")
    section: Optional[str] = Field(None, description="Section/chapter title")
    position: int = Field(..., ge=1, description="Sequential position in the book")
    metadata: Optional[dict] = Field(None, description="Additional metadata (word count, etc.)")


class BookChunkResponse(BaseModel):
    """
    Model for chunk response with embedding omitted for efficiency.
    """
    id: str
    text_content: str
    book_id: str
    page_number: Optional[int]
    section: Optional[str]
    position: int
    metadata: Optional[dict]
    created_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk-123",
                "text_content": "This is a sample chunk of book content that contains important information...",
                "book_id": "book-456",
                "page_number": 25,
                "section": "Chapter 3: Introduction",
                "position": 1,
                "metadata": {"word_count": 150},
                "created_at": "2023-10-01T12:00:00Z"
            }
        }
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.database import Base
import uuid


class Book(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    version = Column(String(50), nullable=False)
    file_type = Column(String(10), default='text')  # pdf, md, txt, etc.
    total_pages = Column(Integer)
    total_chunks = Column(Integer, default=0)  # Number of chunks created
    total_tokens = Column(Integer, default=0)  # Total tokens in the book
    processing_status = Column(String(20), default='pending')  # pending, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    processing_time_seconds = Column(Float)  # Time taken to process the book
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    book_metadata = Column(JSON)  # Additional book metadata (author, publisher, etc.)
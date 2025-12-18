from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.config.database import Base
import uuid


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter = Column(String(200))
    section = Column(String(200))
    page_range = Column(String(50))
    text = Column(Text, nullable=False)
    # Note: embedding column will be handled separately since it requires vector type
    chunk_id = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
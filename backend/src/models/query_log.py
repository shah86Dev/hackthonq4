from sqlalchemy import Column, String, DateTime, Text, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY
from src.config.database import Base
import uuid


class QueryLog(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    mode = Column(String(20), nullable=False)  # 'full-book' or 'selected-text'
    question = Column(Text, nullable=False)
    selected_text = Column(Text)  # User-highlighted text (for selected-text mode)
    retrieved_chunk_ids = Column(ARRAY(UUID(as_uuid=True)))  # List of chunk IDs that were retrieved
    latency = Column(Float)  # Query processing time in seconds
    tokens_used = Column(Integer)  # Number of tokens consumed in the query
    created_at = Column(DateTime(timezone=True), server_default=func.now())
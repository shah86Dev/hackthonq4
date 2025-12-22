from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.database import Base
import uuid


class Answer(Base):
    __tablename__ = "answers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    answer_text = Column(Text, nullable=False)
    citations = Column(JSON, nullable=False)  # Citation information with page/chapter references
    confidence_score = Column(Float, nullable=False)  # Confidence score (0-1) of the answer
    created_at = Column(DateTime(timezone=True), server_default=func.now())
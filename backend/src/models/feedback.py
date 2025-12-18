from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.config.database import Base
import uuid


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    answer_id = Column(UUID(as_uuid=True), ForeignKey("answers.id", ondelete="CASCADE"), nullable=False)
    user_rating = Column(String(10))  # User rating ('positive', 'negative', 'neutral')
    correction = Column(Text)  # User-provided correction if answer was incorrect
    created_at = Column(DateTime(timezone=True), server_default=func.now())
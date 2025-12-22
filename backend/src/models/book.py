from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.database import Base
import uuid


class Book(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    book_metadata = Column(JSON)  # Additional book metadata (author, publisher, etc.)
from sqlalchemy import Column, String, Text, Integer, JSON
from sqlalchemy.orm import relationship
from .base import Base


class ContentChunk(Base):
    __tablename__ = "content_chunks"

    chapter_id = Column(String, nullable=False)  # Reference to chapter
    content = Column(Text, nullable=False)  # Text content for RAG
    content_urdu = Column(Text)  # Urdu translation
    embedding = Column(JSON)  # Vector embedding (stored as JSON array)
    chunk_type = Column(String, default="theory")  # theory, lab, quiz, reference
    metadata = Column(JSON)  # For AI indexing
    source_start_pos = Column(Integer, default=0)  # Start position in source document
    source_end_pos = Column(Integer, default=0)  # End position in source document
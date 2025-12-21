from sqlalchemy import Column, String, Text, Integer, JSON, Boolean
from sqlalchemy.orm import relationship
from .base import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    user_id = Column(Integer, nullable=True)  # Nullable for anonymous users
    session_token = Column(String, nullable=False)  # For anonymous sessions
    messages = Column(JSON)  # Array of message objects
    session_metadata = Column(JSON)  # Additional session metadata


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    session_id = Column(Integer, nullable=False)  # Reference to ChatSession
    sender = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)  # Message content
    source_chunks = Column(JSON)  # Array of content chunk IDs used for response
    is_grounding_valid = Column(Boolean, default=True)  # Whether response is properly grounded in textbook
    message_metadata = Column(JSON)  # Additional message metadata
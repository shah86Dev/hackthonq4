from sqlalchemy import Column, String, Boolean, Text, DateTime, Integer, func
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    __tablename__ = "users"

    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="student")  # student, instructor, admin
    background = Column(Text)  # educational/professional background
    preferred_language = Column(String, default="en")  # en, ur
    personalization_enabled = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Relationships
    chapter_progress = relationship("ChapterProgress", back_populates="user")
    quiz_results = relationship("QuizResult", back_populates="user")
    learning_events = relationship("LearningEvent", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")


class ChapterProgress(Base):
    __tablename__ = "chapter_progress"

    user_id = Column(Integer, nullable=False)
    chapter_id = Column(String, nullable=False)  # Could be a reference to chapter
    status = Column(String, default="not_started")  # not_started, in_progress, completed
    current_position = Column(Integer, default=0)
    time_spent = Column(Integer, default=0)  # in seconds
    completed_at = Column(DateTime, nullable=True)
    personalization_level = Column(Integer, default=0)  # 0-100 scale

    # Relationships
    user = relationship("User", back_populates="chapter_progress")


class QuizResult(Base):
    __tablename__ = "quiz_results"

    user_id = Column(Integer, nullable=False)
    quiz_id = Column(String, nullable=False)  # Could be a reference to quiz
    score = Column(Integer)  # percentage score
    attempts = Column(Integer, default=1)
    # Additional fields for answers would go here

    # Relationships
    user = relationship("User", back_populates="quiz_results")


class LearningEvent(Base):
    __tablename__ = "learning_events"

    user_id = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)  # chapter_view, lab_start, lab_complete, quiz_start, quiz_complete
    entity_id = Column(String, nullable=False)  # ID of the entity involved
    entity_type = Column(String, nullable=False)  # chapter, lab, quiz
    event_metadata = Column(String)  # JSON string for additional event data

    # Relationships
    user = relationship("User", back_populates="learning_events")
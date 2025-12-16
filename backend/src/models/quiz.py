from sqlalchemy import Column, String, Text, Integer, JSON, Boolean
from sqlalchemy.orm import relationship
from .base import Base


class Quiz(Base):
    __tablename__ = "quizzes"

    title = Column(String, nullable=False)
    chapter_id = Column(String, nullable=False)  # Reference to chapter
    questions = Column(JSON)  # Array of question objects
    passing_score = Column(Integer, default=70)  # percentage required to pass
    time_limit = Column(Integer, default=0)  # in minutes, 0 if no limit
    randomize_questions = Column(Boolean, default=True)
    feedback_mode = Column(String, default="immediate")  # immediate, delayed, none
    difficulty_level = Column(String, default="beginner")  # beginner, intermediate, advanced
    metadata = Column(JSON)  # Additional metadata

    # Relationships
    chapter = relationship("Chapter", back_populates="quizzes")


class Question(Base):
    __tablename__ = "questions"

    quiz_id = Column(String, nullable=False)  # Reference to quiz
    question_text = Column(Text, nullable=False)
    question_urdu = Column(Text)  # Urdu translation
    question_type = Column(String, default="multiple_choice")  # multiple_choice, true_false, short_answer, essay
    options = Column(JSON)  # Array of options for multiple choice
    correct_answer = Column(Text)  # Correct answer
    explanation = Column(Text)  # Explanation of correct answer
    explanation_urdu = Column(Text)  # Urdu translation
    difficulty_level = Column(String, default="beginner")  # beginner, intermediate, advanced
    tags = Column(JSON)  # Array of topic tags
    metadata = Column(JSON)  # Additional metadata
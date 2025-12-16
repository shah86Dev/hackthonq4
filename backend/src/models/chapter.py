from sqlalchemy import Column, String, Text, Integer, JSON
from sqlalchemy.orm import relationship
from .base import Base


class Chapter(Base):
    __tablename__ = "chapters"

    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # Markdown content
    content_urdu = Column(Text)  # Urdu translation of content
    module = Column(String, nullable=False)  # ros2, gazebo, isaac, vla
    difficulty_level = Column(String, default="beginner")  # beginner, intermediate, advanced
    learning_objectives = Column(JSON)  # Array of learning objectives
    prerequisites = Column(JSON)  # Array of chapter IDs that are prerequisites
    estimated_reading_time = Column(Integer)  # in minutes
    metadata = Column(JSON)  # For AI indexing

    # Relationships
    lab_exercises = relationship("LabExercise", back_populates="chapter")
    quizzes = relationship("Quiz", back_populates="chapter")


class InstructorResources(Base):
    __tablename__ = "instructor_resources"

    chapter_id = Column(String, nullable=False)  # Reference to chapter
    slides = Column(JSON)  # Array of file paths to slide decks
    slides_urdu = Column(JSON)  # Array of file paths to Urdu slide translations
    assessment_bank = Column(JSON)  # Array of questions for the chapter
    teaching_notes = Column(Text)  # Instructor notes
    teaching_notes_urdu = Column(Text)  # Urdu translation of teaching notes
    lab_setup_guides = Column(JSON)  # Array of file paths to lab setup guides
    lab_setup_guides_urdu = Column(JSON)  # Array of file paths to Urdu lab setup guides


class PersonalizationRule(Base):
    __tablename__ = "personalization_rules"

    trigger_condition = Column(String)  # Condition that triggers personalization
    rule_type = Column(String)  # difficulty_adjustment, example_substitution, content_reordering
    target_content = Column(String)  # ID of content to modify
    adjustment_parameters = Column(JSON)  # Parameters for personalization
    user_background_match = Column(String)  # Background that triggers this rule
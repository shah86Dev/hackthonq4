from sqlalchemy import Column, String, Text, Integer, JSON
from sqlalchemy.orm import relationship
from .base import Base


class LabExercise(Base):
    __tablename__ = "lab_exercises"

    title = Column(String, nullable=False)
    description = Column(Text)
    chapter_id = Column(String, nullable=False)  # Reference to chapter
    simulation_environment = Column(String, nullable=False)  # isaac_sim, gazebo, unity
    ros2_package = Column(String)  # ROS2 package name for the lab
    instructions = Column(Text, nullable=False)
    instructions_urdu = Column(Text)  # Urdu translation
    expected_outcomes = Column(JSON)  # Array of expected outcomes
    difficulty_level = Column(String, default="beginner")  # beginner, intermediate, advanced
    estimated_duration = Column(Integer)  # in minutes
    validation_criteria = Column(JSON)  # Array of validation criteria
    assets = Column(JSON)  # Array of file paths to required assets
    metadata = Column(JSON)  # Additional metadata

    # Relationships
    chapter = relationship("Chapter", back_populates="lab_exercises")
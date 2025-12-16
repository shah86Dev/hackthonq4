from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# User schemas
class UserBase(BaseModel):
    email: str
    name: str
    role: str = "student"
    background: Optional[str] = None
    preferred_language: str = "en"
    personalization_enabled: bool = False


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    background: Optional[str] = None
    preferred_language: Optional[str] = None
    personalization_enabled: Optional[bool] = None


class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True


# Chapter schemas
class ChapterBase(BaseModel):
    title: str
    content: str
    module: str
    difficulty_level: str = "beginner"
    learning_objectives: List[str] = []
    estimated_reading_time: Optional[int] = None


class ChapterCreate(ChapterBase):
    content_urdu: Optional[str] = None


class Chapter(ChapterBase):
    id: int
    content_urdu: Optional[str] = None

    class Config:
        from_attributes = True


# Lab Exercise schemas
class LabExerciseBase(BaseModel):
    title: str
    description: Optional[str] = None
    chapter_id: str
    simulation_environment: str
    instructions: str
    difficulty_level: str = "beginner"
    estimated_duration: Optional[int] = None


class LabExerciseCreate(LabExerciseBase):
    instructions_urdu: Optional[str] = None


class LabExercise(LabExerciseBase):
    id: int
    instructions_urdu: Optional[str] = None

    class Config:
        from_attributes = True


# Quiz schemas
class QuizBase(BaseModel):
    title: str
    chapter_id: str
    passing_score: int = 70
    time_limit: int = 0
    randomize_questions: bool = True
    feedback_mode: str = "immediate"
    difficulty_level: str = "beginner"


class Quiz(QuizBase):
    id: int
    questions: List[dict] = []

    class Config:
        from_attributes = True


# Question schemas
class QuestionBase(BaseModel):
    quiz_id: str
    question_text: str
    question_type: str = "multiple_choice"
    options: List[str] = []
    correct_answer: str
    difficulty_level: str = "beginner"
    tags: List[str] = []


class Question(QuestionBase):
    id: int
    question_urdu: Optional[str] = None
    explanation: Optional[str] = None
    explanation_urdu: Optional[str] = None

    class Config:
        from_attributes = True


# Chat schemas
class ChatMessageBase(BaseModel):
    content: str
    sender: str  # user or assistant


class ChatSessionCreate(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "en"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    source_chunks: List[str]
    is_grounding_valid: bool


# Content Chunk schemas
class ContentChunkBase(BaseModel):
    chapter_id: str
    content: str
    chunk_type: str = "theory"
    source_start_pos: int = 0
    source_end_pos: int = 0


class ContentChunkCreate(ContentChunkBase):
    content_urdu: Optional[str] = None


class ContentChunk(ContentChunkBase):
    id: int
    content_urdu: Optional[str] = None
    embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True
from sqlalchemy.orm import Session
from . import models, schemas
from typing import List


# User CRUD operations
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    from passlib.context import CryptContext
    from datetime import datetime

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password,
        role=user.role,
        background=user.background,
        preferred_language=user.preferred_language,
        personalization_enabled=user.personalization_enabled
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# Chapter CRUD operations
def get_chapter(db: Session, chapter_id: int):
    return db.query(models.Chapter).filter(models.Chapter.id == chapter_id).first()


def get_chapters(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Chapter).offset(skip).limit(limit).all()


def create_chapter(db: Session, chapter: schemas.ChapterCreate):
    db_chapter = models.Chapter(
        title=chapter.title,
        content=chapter.content,
        content_urdu=chapter.content_urdu,
        module=chapter.module,
        difficulty_level=chapter.difficulty_level,
        learning_objectives=chapter.learning_objectives,
        estimated_reading_time=chapter.estimated_reading_time
    )
    db.add(db_chapter)
    db.commit()
    db.refresh(db_chapter)
    return db_chapter


# Lab Exercise CRUD operations
def get_lab_exercise(db: Session, lab_id: int):
    return db.query(models.LabExercise).filter(models.LabExercise.id == lab_id).first()


def get_lab_exercises(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.LabExercise).offset(skip).limit(limit).all()


def create_lab_exercise(db: Session, lab_exercise: schemas.LabExerciseCreate):
    db_lab = models.LabExercise(
        title=lab_exercise.title,
        description=lab_exercise.description,
        chapter_id=lab_exercise.chapter_id,
        simulation_environment=lab_exercise.simulation_environment,
        instructions=lab_exercise.instructions,
        instructions_urdu=lab_exercise.instructions_urdu,
        difficulty_level=lab_exercise.difficulty_level,
        estimated_duration=lab_exercise.estimated_duration
    )
    db.add(db_lab)
    db.commit()
    db.refresh(db_lab)
    return db_lab


# Quiz CRUD operations
def get_quiz(db: Session, quiz_id: int):
    return db.query(models.Quiz).filter(models.Quiz.id == quiz_id).first()


def get_quizzes(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Quiz).offset(skip).limit(limit).all()


def create_quiz(db: Session, quiz: schemas.Quiz):
    db_quiz = models.Quiz(
        title=quiz.title,
        chapter_id=quiz.chapter_id,
        passing_score=quiz.passing_score,
        time_limit=quiz.time_limit,
        randomize_questions=quiz.randomize_questions,
        feedback_mode=quiz.feedback_mode,
        difficulty_level=quiz.difficulty_level,
        questions=quiz.questions
    )
    db.add(db_quiz)
    db.commit()
    db.refresh(db_quiz)
    return db_quiz


# Content Chunk CRUD operations
def get_content_chunk(db: Session, chunk_id: int):
    return db.query(models.ContentChunk).filter(models.ContentChunk.id == chunk_id).first()


def get_content_chunks(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ContentChunk).offset(skip).limit(limit).all()


def create_content_chunk(db: Session, chunk: schemas.ContentChunkCreate):
    db_chunk = models.ContentChunk(
        chapter_id=chunk.chapter_id,
        content=chunk.content,
        content_urdu=chunk.content_urdu,
        chunk_type=chunk.chunk_type,
        source_start_pos=chunk.source_start_pos,
        source_end_pos=chunk.source_end_pos
    )
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk
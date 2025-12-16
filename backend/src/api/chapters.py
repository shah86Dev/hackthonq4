from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import crud, models, schemas
from ..database import get_db
from ..auth import get_current_user

router = APIRouter()


@router.get("/", response_model=List[schemas.Chapter])
def read_chapters(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve chapters with optional pagination
    """
    chapters = crud.get_chapters(db, skip=skip, limit=limit)
    return chapters


@router.get("/{chapter_id}", response_model=schemas.Chapter)
def read_chapter(
    chapter_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve a specific chapter by ID
    """
    chapter = crud.get_chapter(db, chapter_id=chapter_id)
    if chapter is None:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return chapter


@router.post("/", response_model=schemas.Chapter, status_code=status.HTTP_201_CREATED)
def create_chapter(
    chapter: schemas.ChapterCreate,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Create a new chapter
    """
    db_chapter = crud.create_chapter(db, chapter=chapter)
    return db_chapter


@router.put("/{chapter_id}", response_model=schemas.Chapter)
def update_chapter(
    chapter_id: int,
    chapter: schemas.ChapterCreate,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Update an existing chapter
    """
    # For now, we'll implement basic update functionality
    # In a real implementation, we would have an update_chapter function in crud
    existing_chapter = crud.get_chapter(db, chapter_id=chapter_id)
    if existing_chapter is None:
        raise HTTPException(status_code=404, detail="Chapter not found")

    # Update the chapter fields
    for var, value in vars(chapter).items():
        setattr(existing_chapter, var, value)

    db.add(existing_chapter)
    db.commit()
    db.refresh(existing_chapter)

    return existing_chapter


@router.delete("/{chapter_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chapter(
    chapter_id: int,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Delete a chapter
    """
    chapter = crud.get_chapter(db, chapter_id=chapter_id)
    if chapter is None:
        raise HTTPException(status_code=404, detail="Chapter not found")

    db.delete(chapter)
    db.commit()

    return {"message": "Chapter deleted successfully"}
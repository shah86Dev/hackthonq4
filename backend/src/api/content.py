from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import crud, models, schemas
from ..database import get_db
from ..auth import get_current_user

router = APIRouter()


@router.get("/", response_model=List[schemas.ContentChunk])
def read_content_chunks(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve content chunks with optional pagination
    """
    chunks = crud.get_content_chunks(db, skip=skip, limit=limit)
    return chunks


@router.get("/{chunk_id}", response_model=schemas.ContentChunk)
def read_content_chunk(
    chunk_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve a specific content chunk by ID
    """
    chunk = crud.get_content_chunk(db, chunk_id=chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Content chunk not found")
    return chunk


@router.post("/", response_model=schemas.ContentChunk, status_code=status.HTTP_201_CREATED)
def create_content_chunk(
    chunk: schemas.ContentChunkCreate,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Create a new content chunk
    """
    db_chunk = crud.create_content_chunk(db, chunk=chunk)
    return db_chunk


@router.put("/{chunk_id}", response_model=schemas.ContentChunk)
def update_content_chunk(
    chunk_id: int,
    chunk: schemas.ContentChunkCreate,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Update an existing content chunk
    """
    # For now, we'll implement basic update functionality
    existing_chunk = crud.get_content_chunk(db, chunk_id=chunk_id)
    if existing_chunk is None:
        raise HTTPException(status_code=404, detail="Content chunk not found")

    # Update the chunk fields
    for var, value in vars(chunk).items():
        setattr(existing_chunk, var, value)

    db.add(existing_chunk)
    db.commit()
    db.refresh(existing_chunk)

    return existing_chunk


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_content_chunk(
    chunk_id: int,
    db: Session = Depends(get_db)
    # current_user: models.User = Depends(get_current_user)  # Uncomment to require auth
):
    """
    Delete a content chunk
    """
    chunk = crud.get_content_chunk(db, chunk_id=chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Content chunk not found")

    db.delete(chunk)
    db.commit()

    return {"message": "Content chunk deleted successfully"}
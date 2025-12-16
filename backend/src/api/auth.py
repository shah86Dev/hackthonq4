from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from .. import crud, models, schemas
from ..auth import authenticate_user, create_access_token, get_current_user
from ..database import get_db
from ..config import settings

router = APIRouter()


# Define Token schema here since it's not in the main schemas module
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: str
    name: str
    role: str


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role
    }


@router.post("/register", response_model=schemas.User)
async def register_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    """
    # Check if user already exists
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # Create user with hashed password
    from ..auth import create_user_with_hashed_password
    return create_user_with_hashed_password(db, user)


@router.get("/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    """
    Get current user info
    """
    return current_user
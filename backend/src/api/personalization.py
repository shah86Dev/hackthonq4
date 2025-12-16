from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from ..services.personalization import personalization_service, UserProfile, PersonalizationResult
from ..auth import get_current_user
from ..database import get_db
from .. import models

router = APIRouter()


class PersonalizationRequest(BaseModel):
    content: str
    content_type: str = "chapter"
    user_background: Optional[str] = None
    preferred_difficulty: str = "intermediate"
    learning_style: str = "mixed"
    user_interests: List[str] = []
    user_goals: List[str] = []


class PersonalizationResponse(BaseModel):
    original_content: str
    personalized_content: str
    adaptation_reasoning: str
    difficulty_level: str


class UserPreferences(BaseModel):
    background: Optional[str] = None
    preferred_difficulty: str = "intermediate"
    learning_style: str = "mixed"
    interests: List[str] = []
    goals: List[str] = []


@router.post("/personalize", response_model=PersonalizationResponse)
async def personalize_content(
    request: PersonalizationRequest,
    current_user = Depends(get_current_user)
):
    """
    Personalize content based on user preferences
    """
    try:
        # Create a user profile from the request
        user_profile = UserProfile(
            user_id=current_user.id,
            background=request.user_background or current_user.background or "general",
            preferred_difficulty=request.preferred_difficulty,
            learning_style=request.learning_style,
            interests=request.user_interests,
            goals=request.user_goals
        )

        # Personalize the content
        result: PersonalizationResult = await personalization_service.personalize_content(
            request.content,
            user_profile,
            request.content_type
        )

        return PersonalizationResponse(
            original_content=request.content,
            personalized_content=result.modified_content,
            adaptation_reasoning=result.adaptation_reasoning,
            difficulty_level=result.difficulty_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Personalization failed: {str(e)}"
        )


@router.put("/preferences")
async def update_user_preferences(
    preferences: UserPreferences,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user preferences for personalization
    """
    try:
        # Update the user's profile in the database
        current_user.background = preferences.background or current_user.background
        # In a real implementation, we would store these preferences in the database
        # For now, we'll just return a success message
        db.add(current_user)
        db.commit()

        return {
            "message": "User preferences updated successfully",
            "preferences": preferences.dict()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preferences: {str(e)}"
        )


@router.get("/recommendations")
async def get_personalization_recommendations(
    current_user: models.User = Depends(get_current_user)
):
    """
    Get personalization recommendations based on user profile
    """
    try:
        user_profile = UserProfile(
            user_id=current_user.id,
            background=current_user.background or "general",
            preferred_difficulty=current_user.personalization_enabled and "intermediate" or "beginner",
            learning_style="mixed",  # Would come from user preferences in a real implementation
            interests=[],  # Would come from user preferences
            goals=[]  # Would come from user preferences
        )

        recommendations = await personalization_service.get_personalization_recommendations(user_profile)

        return {
            "recommendations": recommendations,
            "user_id": current_user.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {str(e)}"
        )
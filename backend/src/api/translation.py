from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..services.translation import translation_service
from ..auth import get_current_user
from ..database import get_db

router = APIRouter()


class TranslationRequest(BaseModel):
    text: str
    target_language: str = "ur"
    source_language: str = "en"
    context: Optional[str] = None


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    context_used: Optional[str]


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    current_user = Depends(get_current_user)  # Require authentication
):
    """
    Translate text between languages
    Currently supports English to Urdu translation
    """
    try:
        if request.target_language == "ur" and request.source_language == "en":
            translated_text = await translation_service.translate_to_urdu(
                request.text,
                request.context
            )
        elif request.target_language == "en" and request.source_language == "ur":
            translated_text = await translation_service.translate_to_english(
                request.text,
                request.context
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Translation from {request.source_language} to {request.target_language} not supported"
            )

        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            context_used=request.context
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


class BatchTranslationRequest(BaseModel):
    texts: list[str]
    target_language: str = "ur"
    source_language: str = "en"


class BatchTranslationResponse(BaseModel):
    translations: list[TranslationResponse]


@router.post("/translate-batch", response_model=BatchTranslationResponse)
async def translate_batch(
    request: BatchTranslationRequest,
    current_user = Depends(get_current_user)  # Require authentication
):
    """
    Translate multiple texts at once
    """
    try:
        # For now, process one at a time - in production, this would be optimized
        translations = []
        for text in request.texts:
            if request.target_language == "ur" and request.source_language == "en":
                translated_text = await translation_service.translate_to_urdu(text)
            elif request.target_language == "en" and request.source_language == "ur":
                translated_text = await translation_service.translate_to_english(text)
            else:
                translated_text = f"[Translation not supported for {request.source_language} to {request.target_language}]"

            translation_response = TranslationResponse(
                original_text=text,
                translated_text=translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                context_used=None
            )
            translations.append(translation_response)

        return BatchTranslationResponse(translations=translations)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch translation failed: {str(e)}"
        )
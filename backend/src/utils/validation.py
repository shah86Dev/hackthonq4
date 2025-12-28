from pydantic import BaseModel, validator, root_validator, ValidationError
from typing import Optional, List, Dict, Any
import re
import uuid
from enum import Enum
import logging
from functools import wraps
from fastapi import HTTPException, status
from src.utils.error_handlers import AppException, ErrorCode

logger = logging.getLogger(__name__)

# Validation constants
MAX_QUESTION_LENGTH = 1000
MAX_SELECTED_TEXT_LENGTH = 5000
MIN_QUESTION_LENGTH = 3
MIN_SELECTED_TEXT_LENGTH = 5
MAX_TITLE_LENGTH = 500
MAX_AUTHOR_LENGTH = 100
MAX_CONTENT_LENGTH = 1000000  # 1MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = {'pdf', 'md', 'markdown', 'txt'}

class BookFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    MD = "md"
    TXT = "txt"


class QuestionValidator:
    """Validator for question-related inputs"""

    @staticmethod
    def validate_question(question: str) -> str:
        """Validate a question string"""
        if not question:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Question is required"
            )

        if len(question.strip()) < MIN_QUESTION_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Question must be at least {MIN_QUESTION_LENGTH} characters long"
            )

        if len(question) > MAX_QUESTION_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Question must be no more than {MAX_QUESTION_LENGTH} characters long"
            )

        # Check for potentially harmful content
        if QuestionValidator.contains_malicious_content(question):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Question contains potentially harmful content"
            )

        return question.strip()

    @staticmethod
    def contains_malicious_content(text: str) -> bool:
        """Check if text contains potentially malicious patterns"""
        harmful_patterns = [
            r'<script',  # XSS attempts
            r'javascript:',  # JavaScript in URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval() function
            r'exec\s*\(',  # exec() function
        ]

        text_lower = text.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                return True
        return False


class SelectedTextValidator:
    """Validator for selected text inputs"""

    @staticmethod
    def validate_selected_text(selected_text: Optional[str]) -> Optional[str]:
        """Validate selected text context"""
        if selected_text is None:
            return None

        if len(selected_text.strip()) < MIN_SELECTED_TEXT_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Selected text must be at least {MIN_SELECTED_TEXT_LENGTH} characters long"
            )

        if len(selected_text) > MAX_SELECTED_TEXT_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Selected text must be no more than {MAX_SELECTED_TEXT_LENGTH} characters long"
            )

        # Check for potentially harmful content
        if QuestionValidator.contains_malicious_content(selected_text):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Selected text contains potentially harmful content"
            )

        return selected_text.strip()


class BookIdValidator:
    """Validator for book ID inputs"""

    @staticmethod
    def validate_book_id(book_id: str) -> str:
        """Validate book ID is a valid UUID"""
        try:
            uuid.UUID(book_id)
        except ValueError:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Invalid book ID format. Must be a valid UUID."
            )

        return book_id


class ContentValidator:
    """Validator for content-related inputs"""

    @staticmethod
    def validate_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
        """Validate content length and format"""
        if not content:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Content is required"
            )

        if len(content) > max_length:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Content exceeds maximum length of {max_length} characters"
            )

        # Check for potentially harmful content
        if QuestionValidator.contains_malicious_content(content):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Content contains potentially harmful content"
            )

        return content


class FileValidator:
    """Validator for file uploads"""

    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate file size is within limits"""
        return file_size <= MAX_FILE_SIZE

    @staticmethod
    def validate_file_type(filename: str) -> str:
        """Validate file type and return normalized format"""
        if not filename:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Filename is required"
            )

        # Extract file extension
        if '.' not in filename:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="File must have an extension"
            )

        ext = filename.split('.')[-1].lower()

        if ext not in ALLOWED_FILE_TYPES:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"File type '{ext}' is not supported. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
            )

        # Normalize 'md' to 'markdown' for consistency
        if ext == 'md':
            ext = 'markdown'
        elif ext == 'markdown':
            ext = 'markdown'

        return ext

    @staticmethod
    def validate_file_content(content: bytes) -> bool:
        """Basic validation of file content"""
        if not content:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="File content is empty"
            )

        # Check for binary content in text files (simple heuristic)
        try:
            content_str = content.decode('utf-8', errors='replace')
            # If a large portion of the content is replacement characters, it might be binary
            replacement_chars = content_str.count('\ufffd')
            if replacement_chars / len(content_str) > 0.1:  # More than 10% replacement chars
                return False
        except UnicodeDecodeError:
            return False

        return True


class BookMetadataValidator:
    """Validator for book metadata"""

    @staticmethod
    def validate_title(title: str) -> str:
        """Validate book title"""
        if not title or not title.strip():
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Book title is required"
            )

        if len(title) > MAX_TITLE_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Book title must be no more than {MAX_TITLE_LENGTH} characters"
            )

        # Check for potentially harmful content
        if QuestionValidator.contains_malicious_content(title):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Book title contains potentially harmful content"
            )

        return title.strip()

    @staticmethod
    def validate_author(author: Optional[str]) -> Optional[str]:
        """Validate book author"""
        if author is None:
            return None

        if len(author) > MAX_AUTHOR_LENGTH:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Author name must be no more than {MAX_AUTHOR_LENGTH} characters"
            )

        # Check for potentially harmful content
        if QuestionValidator.contains_malicious_content(author):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Author name contains potentially harmful content"
            )

        return author.strip()


class SessionValidator:
    """Validator for session-related inputs"""

    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> Optional[str]:
        """Validate session ID if provided"""
        if session_id is None:
            return None

        # Try to parse as UUID
        try:
            uuid.UUID(session_id)
            return session_id
        except ValueError:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail="Invalid session ID format. Must be a valid UUID."
            )


class LanguageValidator:
    """Validator for language inputs"""

    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi'
    }

    @staticmethod
    def validate_language(language: str) -> str:
        """Validate language code"""
        if not language:
            return 'en'  # Default to English

        lang_code = language.lower()[:2]  # Take first 2 characters as language code

        if lang_code not in LanguageValidator.SUPPORTED_LANGUAGES:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail=f"Language '{lang_code}' is not supported. Supported languages: {', '.join(LanguageValidator.SUPPORTED_LANGUAGES)}"
            )

        return lang_code


class ValidationDecorator:
    """Class containing validation decorators"""

    @staticmethod
    def validate_chat_request(func):
        """Decorator to validate chat request parameters"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract parameters from kwargs or request body
            request = kwargs.get('request')

            if request:
                # Validate question
                if hasattr(request, 'question'):
                    request.question = QuestionValidator.validate_question(request.question)

                # Validate selected text
                if hasattr(request, 'selected_text'):
                    request.selected_text = SelectedTextValidator.validate_selected_text(request.selected_text)

                # Validate book ID
                if hasattr(request, 'book_id'):
                    request.book_id = BookIdValidator.validate_book_id(request.book_id)

                # Validate session ID
                if hasattr(request, 'session_id'):
                    request.session_id = SessionValidator.validate_session_id(request.session_id)

                # Validate language
                if hasattr(request, 'language'):
                    request.language = LanguageValidator.validate_language(request.language)

            return await func(*args, **kwargs)
        return wrapper

    @staticmethod
    def validate_ingest_request(func):
        """Decorator to validate ingestion request parameters"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract parameters from kwargs or request body
            request = kwargs.get('request')

            if request:
                # Validate title
                if hasattr(request, 'title'):
                    request.title = BookMetadataValidator.validate_title(request.title)

                # Validate author
                if hasattr(request, 'author'):
                    request.author = BookMetadataValidator.validate_author(request.author)

                # Validate file format
                if hasattr(request, 'file_format'):
                    if request.file_format not in BookFormat.__members__.values():
                        raise AppException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            error_code=ErrorCode.VALIDATION_ERROR,
                            detail=f"Invalid file format. Supported formats: {', '.join(BookFormat.__members__.values())}"
                        )

                # Validate content
                if hasattr(request, 'content'):
                    request.content = ContentValidator.validate_content(request.content)

            return await func(*args, **kwargs)
        return wrapper


# Create convenience functions for common validation scenarios
def validate_chat_params(question: str, selected_text: Optional[str], book_id: str) -> tuple:
    """
    Validate chat parameters and return cleaned values
    """
    validated_question = QuestionValidator.validate_question(question)
    validated_selected_text = SelectedTextValidator.validate_selected_text(selected_text)
    validated_book_id = BookIdValidator.validate_book_id(book_id)

    return validated_question, validated_selected_text, validated_book_id


def validate_ingestion_params(title: str, author: Optional[str], file_format: str, content: str) -> tuple:
    """
    Validate ingestion parameters and return cleaned values
    """
    validated_title = BookMetadataValidator.validate_title(title)
    validated_author = BookMetadataValidator.validate_author(author)

    if file_format not in BookFormat.__members__.values():
        raise AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.VALIDATION_ERROR,
            detail=f"Invalid file format. Supported formats: {', '.join(BookFormat.__members__.values())}"
        )

    validated_content = ContentValidator.validate_content(content)

    return validated_title, validated_author, file_format, validated_content


def validate_file_upload(filename: str, file_size: int, content: bytes) -> str:
    """
    Validate file upload parameters and return normalized format
    """
    # Validate file type
    file_format = FileValidator.validate_file_type(filename)

    # Validate file size
    if not FileValidator.validate_file_size(file_size):
        raise AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.VALIDATION_ERROR,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )

    # Validate file content
    if not FileValidator.validate_file_content(content):
        raise AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.VALIDATION_ERROR,
            detail="File content is not valid for the specified format"
        )

    return file_format


# Pydantic models with validation
class ChatRequestModel(BaseModel):
    """Pydantic model for chat request with validation"""
    question: str
    selected_text: Optional[str] = None
    book_id: str
    session_id: Optional[str] = None
    language: str = "en"

    @validator('question')
    def validate_question(cls, v):
        return QuestionValidator.validate_question(v)

    @validator('selected_text')
    def validate_selected_text(cls, v):
        return SelectedTextValidator.validate_selected_text(v)

    @validator('book_id')
    def validate_book_id(cls, v):
        return BookIdValidator.validate_book_id(v)

    @validator('session_id', pre=True)
    def validate_session_id(cls, v):
        return SessionValidator.validate_session_id(v)

    @validator('language')
    def validate_language(cls, v):
        return LanguageValidator.validate_language(v)


class IngestRequestModel(BaseModel):
    """Pydantic model for ingestion request with validation"""
    title: str
    author: Optional[str] = None
    file_format: str
    version: str = "1.0"
    content: Optional[str] = None

    @validator('title')
    def validate_title(cls, v):
        return BookMetadataValidator.validate_title(v)

    @validator('author')
    def validate_author(cls, v):
        return BookMetadataValidator.validate_author(v)

    @validator('file_format')
    def validate_file_format(cls, v):
        if v not in BookFormat.__members__.values():
            raise ValueError(f"Invalid file format. Supported formats: {', '.join(BookFormat.__members__.values())}")
        return v

    @validator('content')
    def validate_content(cls, v):
        if v:
            return ContentValidator.validate_content(v)
        return v


logger.info("Validation utilities initialized")
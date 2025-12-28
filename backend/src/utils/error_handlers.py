from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Union
import logging
from enum import Enum
import traceback
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# Set up logging
logger = logging.getLogger(__name__)

class ErrorCode(str, Enum):
    """Enumeration of error codes for the application"""
    # General errors
    GENERAL_ERROR = "GENERAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # Authentication/Authorization errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

    # Processing errors
    PROCESSING_FAILED = "PROCESSING_FAILED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Data errors
    INVALID_DATA_FORMAT = "INVALID_DATA_FORMAT"
    DATA_INTEGRITY_VIOLATION = "DATA_INTEGRITY_VIOLATION"

    # External service errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class AppException(HTTPException):
    """Custom application exception with additional error code and details"""

    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        detail: str = None,
        headers: dict = None,
        **kwargs
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.additional_details = kwargs

    def to_dict(self):
        """Convert the exception to a dictionary for JSON response"""
        return {
            "error_code": self.error_code.value,
            "message": self.detail,
            "status_code": self.status_code,
            "additional_details": self.additional_details
        }


class ErrorDetails:
    """Utility class to create detailed error information"""

    @staticmethod
    def create_error_details(
        error_code: ErrorCode,
        message: str,
        details: dict = None,
        trace: str = None
    ) -> dict:
        """Create a structured error details dictionary"""
        error_info = {
            "error_code": error_code.value,
            "message": message,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

        if details:
            error_info["details"] = details

        if trace:
            error_info["trace"] = trace

        return error_info


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors from Pydantic models"""
    logger.warning(f"Validation error for request {request.url}: {exc}")

    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": ".".join(str(loc) for loc in error['loc']),
            "message": error['msg'],
            "type": error['type']
        })

    error_info = ErrorDetails.create_error_details(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Validation failed",
        details={"errors": error_details}
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_info
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception for request {request.url}: {exc.detail}")

    error_info = ErrorDetails.create_error_details(
        error_code=ErrorCode.GENERAL_ERROR,
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_info
    )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions"""
    logger.error(f"Application exception for request {request.url}: {exc.to_dict()}")

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions"""
    # Log the full traceback
    logger.error(f"Unhandled exception for request {request.url}: {str(exc)}", exc_info=True)

    # In production, don't expose internal error details to the client
    error_info = ErrorDetails.create_error_details(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred. Please try again later."
    )

    # In development, you might want to include more details
    import os
    if os.getenv("ENVIRONMENT") == "development":
        error_info["details"] = {
            "error": str(exc),
            "traceback": traceback.format_exc()
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_info
    )


def register_error_handlers(app):
    """Register all error handlers with the FastAPI app"""
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


def handle_external_service_error(service_name: str, error: Exception) -> AppException:
    """Create an appropriate exception for external service errors"""
    logger.error(f"External service {service_name} error: {str(error)}", exc_info=True)

    return AppException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code=ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        detail=f"The {service_name} service is temporarily unavailable. Please try again later.",
        service=service_name,
        original_error=str(error)
    )


def handle_database_error(error: Exception) -> AppException:
    """Create an appropriate exception for database errors"""
    logger.error(f"Database error: {str(error)}", exc_info=True)

    return AppException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=ErrorCode.INTERNAL_ERROR,
        detail="A database error occurred. Please try again later.",
        original_error=str(error)
    )


def handle_validation_error(field: str, value: any, expected_type: str) -> AppException:
    """Create an appropriate exception for validation errors"""
    logger.warning(f"Validation error: Field '{field}' with value '{value}' does not match expected type '{expected_type}'")

    return AppException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code=ErrorCode.VALIDATION_ERROR,
        detail=f"Invalid value for field '{field}'. Expected {expected_type}, got {type(value).__name__}.",
        field=field,
        value=value,
        expected_type=expected_type
    )


def handle_resource_not_found(resource_type: str, resource_id: str) -> AppException:
    """Create an appropriate exception for resource not found errors"""
    logger.info(f"Resource not found: {resource_type} with ID {resource_id}")

    return AppException(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        detail=f"{resource_type.capitalize()} with ID '{resource_id}' was not found.",
        resource_type=resource_type,
        resource_id=resource_id
    )


def handle_rate_limit_error(limit: int, window: int) -> AppException:
    """Create an appropriate exception for rate limit errors"""
    logger.warning(f"Rate limit exceeded: {limit} requests per {window} seconds")

    return AppException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
        detail=f"Rate limit exceeded. Maximum {limit} requests allowed per {window} seconds.",
        limit=limit,
        window=window
    )


# Custom exception classes for specific use cases
class BookNotFoundException(AppException):
    """Exception raised when a book is not found"""
    def __init__(self, book_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found."
        )


class ChunkNotFoundException(AppException):
    """Exception raised when a chunk is not found"""
    def __init__(self, chunk_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            detail=f"Chunk with ID '{chunk_id}' was not found."
        )


class ProcessingFailedException(AppException):
    """Exception raised when processing fails"""
    def __init__(self, message: str, processing_step: str = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.PROCESSING_FAILED,
            detail=message,
            processing_step=processing_step
        )


class InvalidDataFormatException(AppException):
    """Exception raised when data format is invalid"""
    def __init__(self, message: str, format_type: str = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.INVALID_DATA_FORMAT,
            detail=message,
            format_type=format_type
        )


class ExternalServiceException(AppException):
    """Exception raised when external service is unavailable"""
    def __init__(self, service_name: str, original_error: str = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
            detail=f"The {service_name} service is temporarily unavailable. Please try again later.",
            service=service_name,
            original_error=original_error
        )
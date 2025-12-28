from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any
import time

def customize_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Customize the OpenAPI schema for the application
    """
    if app.openapi_schema:
        return app.openapi_schema

    # Generate the default OpenAPI schema
    openapi_schema = get_openapi(
        title="Book-Embedded RAG Chatbot API",
        version="1.0.0",
        description="""
        # Book-Embedded RAG Chatbot API

        This API provides access to a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about book content.

        ## Features

        - **Question Answering**: Ask questions about book content and get contextually accurate answers
        - **Text Selection Context**: Provide selected text for focused questioning
        - **Book Ingestion**: Upload and process various book formats (PDF, Markdown, TXT)
        - **Citation Support**: Answers include citations to source content
        - **Scalable Processing**: Handles large books efficiently

        ## Authentication

        Some endpoints may require authentication. Include your API key in the Authorization header.

        ## Rate Limiting

        API requests are subject to rate limiting to ensure fair usage and system stability.
        """,
        routes=app.routes,
    )

    # Add custom tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "chat",
            "description": "Endpoints for interacting with the chatbot"
        },
        {
            "name": "ingestion",
            "description": "Endpoints for ingesting and processing books"
        },
        {
            "name": "query",
            "description": "Endpoints for querying book content"
        },
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        }
    ]

    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "support@book-rag-chatbot.com",
        "url": "https://github.com/your-org/book-rag-chatbot"
    }

    # Add license information
    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Book-Embedded RAG Chatbot Documentation",
        "url": "https://book-rag-chatbot.readthedocs.io/"
    }

    # Add custom extensions
    openapi_schema["x-logo"] = {
        "url": "https://example.com/logo.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_api_docs(app: FastAPI):
    """
    Set up custom API documentation with enhanced UI
    """
    # Customize the OpenAPI schema
    app.openapi = lambda: customize_openapi(app)

    # Add custom Swagger UI with enhanced features
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui.css",
            swagger_favicon_url="https://example.com/favicon.ico",
            # Additional configuration options
            swagger_ui_parameters={
                "deepLinking": True,
                "docExpansion": "list",  # Expand operations by default
                "defaultModelsExpandDepth": 1,
                "defaultModelExpandDepth": 1,
                "displayRequestDuration": True,
                "showExtensions": True,
                "showCommonExtensions": True,
                "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"]
            }
        )

    # Add Redoc documentation as alternative
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js",
            redoc_favicon_url="https://example.com/favicon.ico",
        )

    # Add a custom API info endpoint
    @app.get("/api/info", tags=["health"], summary="Get API Information")
    async def get_api_info():
        """
        Get detailed information about the API including version, features, and status.
        """
        import pkg_resources
        try:
            version = pkg_resources.get_distribution("book-rag-chatbot").version
        except:
            version = "1.0.0"

        return {
            "title": app.title,
            "version": app.version,
            "description": app.description,
            "contact": openapi_schema.get("info", {}).get("contact", {}),
            "license": openapi_schema.get("info", {}).get("license", {}),
            "servers": app.servers,
            "features": [
                "RAG-based question answering",
                "Multi-format book ingestion (PDF, MD, TXT)",
                "Context-aware responses",
                "Citation support",
                "Rate limiting",
                "Performance monitoring"
            ],
            "status": "ready",
            "timestamp": int(time.time())
        }

    # Add OpenAPI JSON schema endpoint with custom description
    @app.get(app.openapi_url, include_in_schema=False)
    async def get_open_api_endpoint():
        return customize_openapi(app)


# Example models for documentation purposes
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    """
    question: str = "What is the main theme of this book?"
    selected_text: Optional[str] = None
    book_id: str = "book-12345"
    session_id: Optional[str] = None
    language: str = "en"

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the main theme of this book?",
                "selected_text": "This selected text provides additional context for the question.",
                "book_id": "book-12345",
                "session_id": "session-67890",
                "language": "en"
            }
        }


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint
    """
    response: str
    source_chunks: List[dict]
    session_id: str
    confidence_score: Optional[float] = None
    response_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "response": "The main theme of this book is artificial intelligence and its applications in modern technology...",
                "source_chunks": [
                    {
                        "text": "Artificial intelligence is transforming industries worldwide...",
                        "section": "Chapter 1",
                        "page_range": "1-5"
                    }
                ],
                "session_id": "session-67890",
                "confidence_score": 0.92,
                "response_time_ms": 1200,
                "tokens_used": 150
            }
        }


class IngestRequest(BaseModel):
    """
    Request model for ingestion endpoint
    """
    title: str
    author: Optional[str] = None
    file_format: str  # pdf, md, txt
    version: Optional[str] = "1.0"

    class Config:
        schema_extra = {
            "example": {
                "title": "Introduction to Machine Learning",
                "author": "Jane Doe",
                "file_format": "pdf",
                "version": "1.0"
            }
        }


class IngestResponse(BaseModel):
    """
    Response model for ingestion endpoint
    """
    status: str
    book_id: str
    chunks_processed: int
    file_format: str
    message: str

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "book_id": "book-abc123",
                "chunks_processed": 45,
                "file_format": "pdf",
                "message": "Successfully ingested pdf book 'Introduction to Machine Learning' with 45 chunks"
            }
        }


# Initialize the documentation when the module is imported
def init_docs(app: FastAPI):
    """
    Initialize API documentation for the application
    """
    setup_api_docs(app)
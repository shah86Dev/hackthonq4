"""
Book-Embedded RAG Chatbot - Test Suite

This module contains comprehensive tests for the RAG functionality
including unit tests, integration tests, and end-to-end tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from datetime import datetime
import uuid

from backend.app import app
from src.database import Base, get_db
from src.services.ingestion_service import IngestionService
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.agents.coordinator_agent import CoordinatorAgent
from src.models.book import Book
from src.models.chunk import Chunk
from src.models.query_log import QueryLog


# Create an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Override the get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create a test client
client = TestClient(app)


class TestRAGFunctionality:
    """
    Test suite for the RAG chatbot functionality
    """

    def setup_method(self):
        """Setup method for each test"""
        self.db = TestingSessionLocal()

        # Create mock services
        with patch('src.services.ingestion_service.OpenAI'), \
             patch('src.services.retrieval_service.OpenAI'), \
             patch('src.services.generation_service.OpenAI'), \
             patch('src.services.qdrant_client.QdrantClient'):

            # Create a mock Qdrant service
            from src.services.qdrant_client import QdrantService
            QdrantService._ensure_collection_exists = Mock()
            QdrantService.store_embedding = Mock()
            QdrantService.store_embeddings_batch = Mock()
            QdrantService.search_similar = Mock(return_value=[
                {
                    "id": "mock-chunk-1",
                    "score": 0.8,
                    "payload": {
                        "text": "This is a test chunk with relevant information.",
                        "chapter": "Chapter 1",
                        "section": "Section 1.1",
                        "page_range": "1-5"
                    }
                }
            ])

            # Create instances of services
            self.ingestion_service = IngestionService()
            self.retrieval_service = RetrievalService()
            self.generation_service = GenerationService()
            self.coordinator_agent = CoordinatorAgent()

    def teardown_method(self):
        """Teardown method for each test"""
        self.db.close()

    def test_ingestion_service_initialization(self):
        """Test that the ingestion service initializes correctly"""
        assert self.ingestion_service is not None
        assert hasattr(self.ingestion_service, 'extract_text_from_pdf')
        assert hasattr(self.ingestion_service, 'extract_text_from_markdown')
        assert hasattr(self.ingestion_service, 'chunk_text')
        assert hasattr(self.ingestion_service, 'generate_embedding')

    def test_retrieval_service_initialization(self):
        """Test that the retrieval service initializes correctly"""
        assert self.retrieval_service is not None
        assert hasattr(self.retrieval_service, 'retrieve_chunks_full_book')
        assert hasattr(self.retrieval_service, 'retrieve_chunks_with_selected_context')

    def test_generation_service_initialization(self):
        """Test that the generation service initializes correctly"""
        assert self.generation_service is not None
        assert hasattr(self.generation_service, 'generate_answer')

    def test_coordinator_agent_initialization(self):
        """Test that the coordinator agent initializes correctly"""
        assert self.coordinator_agent is not None
        assert hasattr(self.coordinator_agent, 'process_query')
        assert hasattr(self.coordinator_agent, 'process_query_with_validation')

    def test_process_query_basic(self):
        """Test basic query processing"""
        # Create a mock book in the database
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the generation service to return a predictable response
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": "This is a test answer based on the book content.",
                "tokens_used": 10,
                "confidence_score": 0.9,
                "response_time_ms": 100,
                "context_chunks_used": [
                    {
                        "id": "mock-chunk-1",
                        "section": "Section 1.1",
                        "page_range": "1-5",
                        "score": 0.8
                    }
                ]
            }

            # Test the coordinator agent with the mocked services
            result = self.coordinator_agent.process_query(
                db=self.db,
                book_id=book_id,
                question="What is this book about?",
                selected_text=None
            )

            # Verify the result structure
            assert "answer" in result
            assert "mode" in result
            assert "book_id" in result
            assert "question" in result
            assert result["book_id"] == book_id
            assert result["question"] == "What is this book about?"
            assert result["mode"] == "full-book"
            assert "confidence_score" in result
            assert "response_time_ms" in result

    def test_process_query_with_selected_text(self):
        """Test query processing with selected text context"""
        # Create a mock book in the database
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the generation service to return a predictable response
        selected_text = "This is the selected text that the user highlighted."
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": f"This is an answer based on the selected text: {selected_text}",
                "tokens_used": 12,
                "confidence_score": 0.95,
                "response_time_ms": 120,
                "context_chunks_used": [
                    {
                        "id": "selected-text",
                        "section": "Selected Text",
                        "page_range": "N/A",
                        "score": 1.0
                    }
                ]
            }

            # Test the coordinator agent with selected text
            result = self.coordinator_agent.process_query(
                db=self.db,
                book_id=book_id,
                question="What does this selected text mean?",
                selected_text=selected_text
            )

            # Verify the result structure
            assert "answer" in result
            assert "mode" in result
            assert "book_id" in result
            assert "question" in result
            assert "selected_text" in result
            assert result["book_id"] == book_id
            assert result["question"] == "What does this selected text mean?"
            assert result["selected_text"] == selected_text
            assert result["mode"] == "selected-text"
            assert "confidence_score" in result
            assert "response_time_ms" in result

    def test_api_chat_endpoint_basic(self):
        """Test the API chat endpoint with a basic query"""
        # Create a mock book in the database
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the services to avoid external dependencies
        with patch('src.services.retrieval_service.RetrievalService') as mock_retrieval, \
             patch('src.services.generation_service.GenerationService') as mock_generation:

            # Configure mock retrieval service
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve_chunks_with_selected_context.return_value = [
                {
                    "id": "mock-chunk-1",
                    "text": "This is a test chunk with relevant information.",
                    "chapter": "Chapter 1",
                    "section": "Section 1.1",
                    "page_range": "1-5",
                    "score": 0.8
                }
            ]
            mock_retrieval.return_value = mock_retrieval_instance

            # Configure mock generation service
            mock_generation_instance = Mock()
            mock_generation_instance.generate_answer.return_value = {
                "answer_text": "This is a test answer based on the book content.",
                "tokens_used": 10,
                "confidence_score": 0.9,
                "response_time_ms": 100,
                "context_chunks_used": [
                    {
                        "id": "mock-chunk-1",
                        "section": "Section 1.1",
                        "page_range": "1-5",
                        "score": 0.8
                    }
                ]
            }
            mock_generation.return_value = mock_generation_instance

            # Make a request to the API
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What is this book about?",
                    "book_id": book_id
                }
            )

            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "source_chunks" in data
            assert "session_id" in data

    def test_api_chat_endpoint_with_selected_text(self):
        """Test the API chat endpoint with selected text"""
        # Create a mock book in the database
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the services to avoid external dependencies
        with patch('src.services.retrieval_service.RetrievalService') as mock_retrieval, \
             patch('src.services.generation_service.GenerationService') as mock_generation:

            # Configure mock retrieval service
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve_chunks_with_selected_context.return_value = [
                {
                    "id": "selected-text",
                    "text": "This is the selected text.",
                    "chapter": "Selected Text",
                    "section": "User Selection",
                    "page_range": "N/A",
                    "score": 1.0
                }
            ]
            mock_retrieval.return_value = mock_retrieval_instance

            # Configure mock generation service
            mock_generation_instance = Mock()
            mock_generation_instance.generate_answer.return_value = {
                "answer_text": "This is an answer based on the selected text.",
                "tokens_used": 8,
                "confidence_score": 0.95,
                "response_time_ms": 80,
                "context_chunks_used": [
                    {
                        "id": "selected-text",
                        "section": "User Selection",
                        "page_range": "N/A",
                        "score": 1.0
                    }
                ]
            }
            mock_generation.return_value = mock_generation_instance

            # Make a request to the API with selected text
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What does this selected text mean?",
                    "selected_text": "This is the selected text.",
                    "book_id": book_id
                }
            )

            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "source_chunks" in data
            assert "session_id" in data

    def test_api_chat_endpoint_invalid_book_id(self):
        """Test the API chat endpoint with an invalid book ID"""
        # Make a request to the API with an invalid book ID
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is this book about?",
                "book_id": "invalid-book-id"
            }
        )

        # Verify that it returns a 400 error
        assert response.status_code == 400
        assert "Invalid book_id" in response.json()["detail"]

    def test_api_chat_endpoint_missing_book(self):
        """Test the API chat endpoint with a book ID that doesn't exist"""
        # Use a valid UUID format but one that doesn't exist in the DB
        fake_book_id = str(uuid.uuid4())

        # Make a request to the API with a non-existent book ID
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is this book about?",
                "book_id": fake_book_id
            }
        )

        # Verify that it returns a 404 error
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_api_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert "Book-Embedded RAG Chatbot Backend" in data["service"]

    def test_api_config_endpoint(self):
        """Test the configuration endpoint"""
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "debug" in data
        assert "qdrant_host" in data
        assert "embedding_model" in data


class TestIngestionService:
    """
    Test suite for the ingestion service
    """

    def test_chunk_text_basic(self):
        """Test basic text chunking functionality"""
        service = IngestionService()
        text = "This is a test sentence. " * 100  # Create a longer text
        chunks = service.chunk_text(text, chunk_size=100, overlap=20)

        # Verify chunks were created
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_detect_format(self):
        """Test format detection functionality"""
        service = IngestionService()

        # Test PDF detection
        assert service.detect_format(b"", "test.pdf") == "pdf"
        assert service.detect_format(b"", "test.PDF") == "pdf"

        # Test Markdown detection
        assert service.detect_format(b"", "test.md") == "markdown"
        assert service.detect_format(b"", "test.markdown") == "markdown"
        assert service.detect_format(b"", "test.MD") == "markdown"

        # Test TXT detection
        assert service.detect_format(b"", "test.txt") == "txt"
        assert service.detect_format(b"", "test.TXT") == "txt"


class TestIntegration:
    """
    Integration tests for the complete system
    """

    def setup_method(self):
        """Setup method for integration tests"""
        self.db = TestingSessionLocal()

    def teardown_method(self):
        """Teardown method for integration tests"""
        self.db.close()

    def test_complete_ingestion_and_query_flow(self):
        """Test the complete flow from ingestion to query"""
        # Mock the Qdrant service
        with patch('src.services.qdrant_client.QdrantService') as mock_qdrant:
            mock_qdrant_instance = Mock()
            mock_qdrant_instance.store_embedding = Mock()
            mock_qdrant_instance.search_similar = Mock(return_value=[
                {
                    "id": "test-chunk-1",
                    "score": 0.9,
                    "payload": {
                        "text": "This is a test chunk for integration.",
                        "chapter": "Chapter 1",
                        "section": "Section 1.1",
                        "page_range": "1-5"
                    }
                }
            ])
            mock_qdrant.return_value = mock_qdrant_instance

            # Create a mock book
            book_id = str(uuid.uuid4())
            book = Book(
                id=book_id,
                title="Integration Test Book",
                version="1.0"
            )
            self.db.add(book)
            self.db.commit()

            # Mock the generation service
            with patch('src.services.generation_service.GenerationService') as mock_generation:
                mock_generation_instance = Mock()
                mock_generation_instance.generate_answer.return_value = {
                    "answer_text": "This is an integration test answer.",
                    "tokens_used": 8,
                    "confidence_score": 0.9,
                    "response_time_ms": 150,
                    "context_chunks_used": [
                        {
                            "id": "test-chunk-1",
                            "section": "Section 1.1",
                            "page_range": "1-5",
                            "score": 0.9
                        }
                    ]
                }
                mock_generation.return_value = mock_generation_instance

                # Test the coordinator agent
                coordinator = CoordinatorAgent()
                result = coordinator.process_query(
                    db=self.db,
                    book_id=book_id,
                    question="What is this book about?",
                    selected_text=None
                )

                # Verify the result
                assert "answer" in result
                assert result["answer"] == "This is an integration test answer."
                assert result["book_id"] == book_id
                assert result["question"] == "What is this book about?"


def run_tests():
    """
    Function to run all tests
    """
    pytest.main([
        "-v",
        __file__,
        "--tb=short"
    ])


if __name__ == "__main__":
    run_tests()
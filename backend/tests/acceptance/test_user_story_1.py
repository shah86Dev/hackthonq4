import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from src.main import app
from src.database import Base, get_db
from src.services.ingestion_service import IngestionService
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.agents.coordinator_agent import CoordinatorAgent
import uuid
from src.models.book import Book
from src.models.chunk import Chunk

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

class TestUserStory1Acceptance:
    """
    Acceptance tests for User Story 1: Query Book Content via Chatbot
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
                        "text": "This is a test chunk with relevant information about the book content.",
                        "chapter": "Chapter 1",
                        "section": "Section 1.1",
                        "page_range": "1-5"
                    }
                },
                {
                    "id": "mock-chunk-2",
                    "score": 0.75,
                    "payload": {
                        "text": "Additional information about the book that might be relevant to the query.",
                        "chapter": "Chapter 1",
                        "section": "Section 1.2",
                        "page_range": "6-10"
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

    def test_acceptance_scenario_1_book_content_query(self):
        """
        Test acceptance scenario: Given a book has been processed and embedded in the system,
        When a user submits a question via the chat interface,
        Then the system returns an answer based on the book content with proper context citations
        """
        # Given: A book has been processed and embedded in the system
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book on Software Engineering",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the generation service to return an answer based on book content
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": "Based on the book content, software engineering involves systematic approaches to developing software systems. The book mentions that it includes requirements engineering, design, implementation, testing, and maintenance as key phases.",
                "tokens_used": 35,
                "confidence_score": 0.88,
                "response_time_ms": 150,
                "context_chunks_used": [
                    {
                        "id": "mock-chunk-1",
                        "section": "Section 1.1",
                        "page_range": "1-5",
                        "score": 0.8
                    },
                    {
                        "id": "mock-chunk-2",
                        "section": "Section 1.2",
                        "page_range": "6-10",
                        "score": 0.75
                    }
                ]
            }

            # When: A user submits a question via the chat interface
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What is software engineering according to this book?",
                    "book_id": book_id
                }
            )

            # Then: The system returns an answer based on the book content with proper context citations
            assert response.status_code == 200
            data = response.json()

            # Verify the response contains an answer
            assert "response" in data
            answer = data["response"]
            assert "software engineering" in answer.lower()
            assert "book content" in answer.lower() or "according to the book" in answer.lower()

            # Verify citations are provided
            assert "source_chunks" in data
            source_chunks = data["source_chunks"]
            assert len(source_chunks) >= 1  # Should have at least one citation

            # Verify session ID is provided
            assert "session_id" in data

    def test_acceptance_scenario_2_no_hallucinations(self):
        """
        Test acceptance scenario: Given a user has asked a question about the book content,
        When the system processes the query,
        Then the response is grounded in the book's content without hallucinations
        """
        # Given: A book has been processed and embedded in the system
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Introduction to Python Programming",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the generation service to return a grounded response
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": "According to the book, Python is a high-level programming language. The book states that it was created by Guido van Rossum and was first released in 1991. The text mentions that Python emphasizes code readability with its notable use of significant whitespace.",
                "tokens_used": 42,
                "confidence_score": 0.92,
                "response_time_ms": 180,
                "context_chunks_used": [
                    {
                        "id": "mock-chunk-1",
                        "section": "Section 1.1",
                        "page_range": "1-5",
                        "score": 0.8
                    }
                ]
            }

            # When: A user asks a question about the book content
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "Who created Python and when was it first released?",
                    "book_id": book_id
                }
            )

            # Then: The response is grounded in the book's content without hallucinations
            assert response.status_code == 200
            data = response.json()

            # Verify the response contains information that could be found in the book
            answer = data["response"]
            assert "guido van rossum" in answer.lower()
            assert "1991" in answer
            assert "high-level programming language" in answer.lower()

            # The response should not contain information not mentioned in the book context
            # This would be verified by ensuring the response is based on the provided context
            assert "book" in answer.lower() or "text" in answer.lower()

            # Verify citations are provided
            assert "source_chunks" in data
            source_chunks = data["source_chunks"]
            assert len(source_chunks) >= 1

    def test_acceptance_scenario_3_no_content_found(self):
        """
        Test scenario where the query doesn't match any book content
        """
        # Given: A book has been processed and embedded in the system
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Test Book",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the retrieval service to return no relevant chunks
        with patch('src.services.retrieval_service.RetrievalService') as mock_retrieval, \
             patch('src.services.generation_service.GenerationService') as mock_generation:

            # Configure mock retrieval service to return no results
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve_chunks_with_selected_context.return_value = []
            mock_retrieval.return_value = mock_retrieval_instance

            # Configure mock generation service to return a response indicating no content found
            mock_generation_instance = Mock()
            mock_generation_instance.generate_answer.return_value = {
                "answer_text": "I cannot find this information in the book.",
                "tokens_used": 8,
                "confidence_score": 0.1,
                "response_time_ms": 50,
                "context_chunks_used": []
            }
            mock_generation.return_value = mock_generation_instance

            # When: A user asks a question not covered by the book
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What is the capital of Mars?",
                    "book_id": book_id
                }
            )

            # Then: The system should acknowledge that the information is not in the book
            assert response.status_code == 200
            data = response.json()

            answer = data["response"]
            assert "cannot find" in answer.lower() or "not in the book" in answer.lower()
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

class TestUserStory2Acceptance:
    """
    Acceptance tests for User Story 2: Query Selected Text Context
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
                        "text": "This is the selected text that the user highlighted in the book. This text is very important for understanding the concept.",
                        "chapter": "Chapter 1",
                        "section": "Section 1.1",
                        "page_range": "1-5"
                    }
                },
                {
                    "id": "mock-chunk-2",
                    "score": 0.75,
                    "payload": {
                        "text": "Additional context about the selected text that might be relevant.",
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

    def test_acceptance_scenario_1_selected_text_context(self):
        """
        Test acceptance scenario: Given a reader has selected text in the book viewer,
        When they ask a question,
        Then the system uses the selected text as primary context for the answer
        """
        # Given: A book has been processed and a text selection is made
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Advanced Programming Concepts",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Selected text from the book
        selected_text = "This is the selected text that the user highlighted in the book. This text is very important for understanding the concept."

        # Mock the generation service to return an answer focused on the selected text
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": f"Based on the selected text: '{selected_text[:50]}...', the concept refers to important programming principles. The selected text emphasizes that this is crucial for understanding the topic.",
                "tokens_used": 25,
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

            # When: The user asks a question with selected text context
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What does this selected text mean?",
                    "selected_text": selected_text,
                    "book_id": book_id
                }
            )

            # Then: The system uses the selected text as primary context
            assert response.status_code == 200
            data = response.json()

            # Verify the response contains information related to the selected text
            answer = data["response"]
            assert "selected text" in answer.lower() or "highlighted" in answer.lower()
            assert selected_text[:20] in answer or "concept" in answer.lower()

            # Verify that the mode is selected-text
            # We can't directly verify this in the current response format, but we can check
            # that the answer is contextually related to the selected text

    def test_acceptance_scenario_2_focused_answer(self):
        """
        Test acceptance scenario: Given a reader has selected text and asked a related question,
        When the system generates the response,
        Then the answer is more focused on the selected content than general book knowledge
        """
        # Given: A book has been processed and a specific text is selected
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Introduction to Machine Learning",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        # Selected text that contains specific information
        selected_text = "Gradient descent is an optimization algorithm used to minimize the cost function by iteratively moving in the direction of steepest descent."

        # Mock the generation service to return a focused answer
        with patch.object(self.generation_service, 'generate_answer') as mock_gen:
            mock_gen.return_value = {
                "answer_text": f"The selected text explains that {selected_text.lower()}. This algorithm is fundamental in machine learning for training models.",
                "tokens_used": 30,
                "confidence_score": 0.92,
                "response_time_ms": 140,
                "context_chunks_used": [
                    {
                        "id": "selected-text",
                        "section": "Selected Text",
                        "page_range": "N/A",
                        "score": 1.0
                    }
                ]
            }

            # When: The user asks a question related to the selected text
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "Explain gradient descent based on this text",
                    "selected_text": selected_text,
                    "book_id": book_id
                }
            )

            # Then: The answer is more focused on the selected content
            assert response.status_code == 200
            data = response.json()

            answer = data["response"]
            # The answer should be specifically about gradient descent as described in the selected text
            assert "gradient descent" in answer.lower()
            assert "optimization algorithm" in answer.lower()
            assert "cost function" in answer.lower()
            assert "steepest descent" in answer.lower()

            # The answer should reference the selected content specifically
            assert "selected text" in answer.lower() or "text explains" in answer.lower()

    def test_selected_text_over_full_book_retrieval(self):
        """
        Test that when selected text is provided, it takes priority over full-book retrieval
        """
        # Given: A book has been processed
        book_id = str(uuid.uuid4())
        book = Book(
            id=book_id,
            title="Programming Fundamentals",
            version="1.0"
        )
        self.db.add(book)
        self.db.commit()

        selected_text = "Object-oriented programming is a programming paradigm based on the concept of objects."

        # Mock services to verify that selected text path is taken
        with patch('src.services.retrieval_service.RetrievalService') as mock_retrieval, \
             patch('src.services.generation_service.GenerationService') as mock_generation:

            # Configure mock retrieval service to track which method is called
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve_chunks_with_selected_context.return_value = [
                {
                    "id": "selected-text",
                    "text": selected_text,
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
                "answer_text": f"The selected text describes object-oriented programming as: {selected_text}",
                "tokens_used": 18,
                "confidence_score": 0.94,
                "response_time_ms": 110,
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

            # When: A query is made with selected text
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What is object-oriented programming?",
                    "selected_text": selected_text,
                    "book_id": book_id
                }
            )

            # Then: Verify the selected text retrieval method was called
            assert response.status_code == 200
            mock_retrieval_instance.retrieve_chunks_with_selected_context.assert_called_once()

            # Verify the response contains the expected answer
            data = response.json()
            assert selected_text[:20] in data["response"]
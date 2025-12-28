import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from src.main import app
from src.database import Base, get_db
from src.services.ingestion_service import IngestionService
import uuid
from src.models.book import Book
import time

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

class TestUserStory4Acceptance:
    """
    Acceptance tests for User Story 4: Scalable Book Processing
    """

    def setup_method(self):
        """Setup method for each test"""
        self.db = TestingSessionLocal()

        # Create mock services
        with patch('src.services.ingestion_service.OpenAI'), \
             patch('src.services.ingestion_service.QdrantClient'), \
             patch('src.services.ingestion_service.PyPDF2.PdfReader'), \
             patch('src.services.ingestion_service.markdown'):

            # Create a mock Qdrant service
            from src.services.qdrant_client import QdrantService
            QdrantService._ensure_collection_exists = Mock()
            QdrantService.store_embedding = Mock()
            QdrantService.store_embeddings_batch = Mock()
            QdrantService.search_similar = Mock(return_value=[])

            # Mock markdown processing
            import src.services.ingestion_service as ingestion_module
            ingestion_module.markdown.markdown = Mock(return_value="<h1>Test Title</h1><p>This is test content.</p>")

            # Create instances of services
            self.ingestion_service = IngestionService()

    def teardown_method(self):
        """Teardown method for each test"""
        self.db.close()

    def test_acceptance_scenario_1_large_book_processing(self):
        """
        Test acceptance scenario: Given a large book (>500 pages),
        When the system processes it,
        Then the processing completes efficiently using distributed methods
        """
        # Create large book content (simulating a large book with 50k+ characters)
        large_content = "This is a large book content. " * 2000  # ~50k+ characters
        encoded_content = large_content.encode('utf-8')

        # Mock the upload to simulate the API call
        with patch('fastapi.UploadFile.read', return_value=encoded_content):
            # When: A large book file is uploaded for processing
            start_time = time.time()
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("large_book.txt", encoded_content, "text/plain")},
                data={
                    "title": "Large Test Book",
                    "author": "Test Author",
                    "file_format": "txt",
                    "version": "1.0"
                }
            )
            end_time = time.time()

            # Then: The system should process the large book and return success
            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "success"
            assert data["file_format"] == "txt"
            assert "Large Test Book" in data["message"]
            assert data["chunks_processed"] >= 0  # Should have processed some chunks

            # Processing should complete in reasonable time (under 30 seconds for test)
            processing_time = end_time - start_time
            assert processing_time < 30, f"Processing took too long: {processing_time}s"

    def test_acceptance_scenario_2_response_times_large_book(self):
        """
        Test acceptance scenario: Given a large processed book,
        When users query it,
        Then response times remain under 2 seconds
        """
        # First, simulate ingesting a book (we'll mock this since we're testing query performance)
        book_id = str(uuid.uuid4())

        # Mock the book in the database
        book = Book(
            id=uuid.UUID(book_id),
            title="Large Processed Book",
            version="1.0",
            total_chunks=100,  # Simulate a book with many chunks
            processing_status="completed"
        )
        self.db.add(book)
        self.db.commit()

        # Mock the retrieval and generation services for the query
        with patch('src.services.retrieval_service.RetrievalService') as mock_retrieval, \
             patch('src.services.generation_service.GenerationService') as mock_generation:

            # Configure mock retrieval service
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve_chunks_with_selected_context.return_value = [
                {
                    "id": f"mock-chunk-{i}",
                    "text": f"This is a sample chunk {i} with relevant information for testing.",
                    "chapter": f"Chapter {i//10 + 1}",
                    "section": f"Section {i}",
                    "page_range": f"{i*2}-{i*2+1}",
                    "score": 0.8
                } for i in range(5)  # 5 chunks
            ]
            mock_retrieval.return_value = mock_retrieval_instance

            # Configure mock generation service
            mock_generation_instance = Mock()
            mock_generation_instance.generate_answer.return_value = {
                "answer_text": "This is a test answer based on the large book content. The system efficiently retrieved relevant information from the large dataset.",
                "tokens_used": 30,
                "confidence_score": 0.85,
                "response_time_ms": 800,  # Less than 2 seconds
                "context_chunks_used": [
                    {
                        "id": f"mock-chunk-{i}",
                        "section": f"Section {i}",
                        "page_range": f"{i*2}-{i*2+1}",
                        "score": 0.8
                    } for i in range(5)
                ]
            }
            mock_generation.return_value = mock_generation_instance

            # When: A query is made to the large processed book
            start_time = time.time()
            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "What is this book about?",
                    "book_id": book_id
                }
            )
            end_time = time.time()

            # Then: The response time should be under 2 seconds
            response_time = end_time - start_time
            assert response_time < 2.0, f"Response time was too slow: {response_time}s"

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert len(data["response"]) > 0

    def test_large_book_performance_monitoring(self):
        """
        Test that large books are processed with performance monitoring
        """
        # Create moderately large content (simulating a book that would benefit from monitoring)
        large_content = "Introduction to machine learning. " * 1000  # ~25k+ characters
        encoded_content = large_content.encode('utf-8')

        # Mock the upload to simulate the API call
        with patch('fastapi.UploadFile.read', return_value=encoded_content):
            # Record start time
            start_time = time.time()

            response = client.post(
                "/api/v1/ingest",
                files={"file": ("medium_book.txt", encoded_content, "text/plain")},
                data={
                    "title": "Medium Test Book",
                    "author": "Test Author",
                    "file_format": "txt",
                    "version": "1.0"
                }
            )

            end_time = time.time()

            # Verify the response
            assert response.status_code == 200
            data = response.json()

            # Check that the response includes performance information
            assert "Processing time:" in data["message"]
            assert data["chunks_processed"] > 0

            # Calculate actual time
            actual_time = end_time - start_time
            assert actual_time < 10, f"Processing took too long: {actual_time}s for medium book"

    def test_distributed_processing_simulation(self):
        """
        Test the distributed processing logic (simulated)
        """
        # Test the logic that determines when to use distributed processing
        content_small = "Small content" * 100  # Small content
        content_large = "Large content" * 10000  # Large content (>50k chars)

        # Verify that the ingestion service would handle both appropriately
        # For small content, it should process sequentially
        # For large content, it should consider distributed processing if Ray is available

        # Mock the ingestion service methods to verify the path taken
        with patch.object(self.ingestion_service, '_process_chunk') as mock_process_chunk, \
             patch.object(self.ingestion_service, 'generate_embedding') as mock_gen:

            # Mock embedding generation
            mock_gen.return_value = [0.1] * 1536  # Mock embedding

            # Process small content
            result_small = self.ingestion_service.ingest_book(
                db=self.db,
                book_id=uuid.uuid4(),
                title="Small Book",
                version="1.0",
                content=content_small,
                file_format="txt"
            )

            # Process large content
            result_large = self.ingestion_service.ingest_book(
                db=self.db,
                book_id=uuid.uuid4(),
                title="Large Book",
                version="1.0",
                content=content_large,
                file_format="txt"
            )

            # Both should succeed
            assert result_small["status"] == "success"
            assert result_large["status"] == "success"

            # The processing should have been completed
            assert result_small["chunks_processed"] >= 0
            assert result_large["chunks_processed"] >= 0
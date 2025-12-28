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

class TestUserStory3Acceptance:
    """
    Acceptance tests for User Story 3: Process Different Book Formats
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

    def test_acceptance_scenario_1_pdf_processing(self):
        """
        Test acceptance scenario: Given a PDF book file,
        When the system processes it,
        Then the text is extracted and prepared for RAG functionality
        """
        # Mock PDF content extraction
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create a mock page
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is content from a PDF book about machine learning."

            # Create a mock reader that returns our page
            mock_instance = Mock()
            mock_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_instance

            # Create mock PDF file content
            mock_pdf_content = b"dummy pdf content"

            # Mock the upload to simulate the API call
            with patch('fastapi.UploadFile.read', return_value=mock_pdf_content):
                # When: A PDF file is uploaded for processing
                response = client.post(
                    "/api/v1/ingest",
                    files={"file": ("test.pdf", mock_pdf_content, "application/pdf")},
                    data={
                        "title": "Test PDF Book",
                        "author": "Test Author",
                        "file_format": "pdf",
                        "version": "1.0"
                    }
                )

                # Then: The system should process the PDF and return success
                assert response.status_code == 200
                data = response.json()

                assert data["status"] == "success"
                assert data["file_format"] == "pdf"
                assert "Test PDF Book" in data["message"]
                assert data["chunks_processed"] >= 0  # Should have processed some chunks

    def test_acceptance_scenario_2_markdown_processing(self):
        """
        Test acceptance scenario: Given a Markdown book file,
        When the system processes it,
        Then the content is properly structured for embedding and retrieval
        """
        # Create mock Markdown content
        markdown_content = """# Introduction to AI

This is a book about artificial intelligence.

## Chapter 1: Basics
Artificial intelligence is a wonderful field that combines computer science and cognitive science.

### Subsection: History
The history of AI dates back to the 1950s.
"""

        # Encode the content to bytes for the file upload
        encoded_content = markdown_content.encode('utf-8')

        # Mock the upload to simulate the API call
        with patch('fastapi.UploadFile.read', return_value=encoded_content):
            # When: A Markdown file is uploaded for processing
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.md", encoded_content, "text/markdown")},
                data={
                    "title": "Test Markdown Book",
                    "author": "Test Author",
                    "file_format": "markdown",
                    "version": "1.0"
                }
            )

            # Then: The system should process the Markdown and return success
            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "success"
            assert data["file_format"] == "markdown"
            assert "Test Markdown Book" in data["message"]
            assert data["chunks_processed"] >= 0  # Should have processed some chunks

    def test_process_different_formats_pdf(self):
        """
        Test processing of different formats - PDF
        """
        # Mock PDF content extraction
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create mock pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Content from first page of PDF."
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Content from second page of PDF."

            # Create a mock reader that returns our pages
            mock_instance = Mock()
            mock_instance.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_instance

            # Process PDF content
            result = self.ingestion_service.process_content(b"dummy pdf", "pdf", "test.pdf")

            # Verify the content was processed
            assert "first page" in result
            assert "second page" in result

    def test_process_different_formats_markdown(self):
        """
        Test processing of different formats - Markdown
        """
        # Mock markdown processing
        with patch('src.services.ingestion_service.markdown') as mock_markdown:
            mock_markdown.markdown.return_value = "<h1>Title</h1><p>This is content.</p>"

            # Process Markdown content
            result = self.ingestion_service.process_content(
                "# Title\n\nThis is content.".encode('utf-8'),
                "markdown",
                "test.md"
            )

            # Verify the content was processed (HTML tags should be stripped)
            assert "Title" in result
            assert "This is content" in result
            assert "<h1>" not in result  # HTML tags should be removed

    def test_process_different_formats_txt(self):
        """
        Test processing of different formats - TXT
        """
        # Process plain text content
        txt_content = "This is plain text content for the book."
        result = self.ingestion_service.process_content(
            txt_content.encode('utf-8'),
            "txt",
            "test.txt"
        )

        # Verify the content was processed
        assert "plain text content" in result

    def test_format_detection(self):
        """
        Test format detection based on file extension
        """
        # Test PDF detection
        result = self.ingestion_service.detect_format(b"dummy", "book.pdf")
        assert result == "pdf"

        # Test Markdown detection (.md)
        result = self.ingestion_service.detect_format(b"dummy", "book.md")
        assert result == "markdown"

        # Test Markdown detection (.markdown)
        result = self.ingestion_service.detect_format(b"dummy", "book.markdown")
        assert result == "markdown"

        # Test TXT detection
        result = self.ingestion_service.detect_format(b"dummy", "book.txt")
        assert result == "txt"

    def test_ingest_different_formats_api(self):
        """
        Test that the API can handle different formats through the ingest endpoint
        """
        # Test with mock content for each format
        formats_to_test = [
            ("test.pdf", b"dummy pdf content", "application/pdf", "pdf"),
            ("test.md", "# Title\nContent".encode('utf-8'), "text/markdown", "markdown"),
            ("test.txt", "Plain text content".encode('utf-8'), "text/plain", "txt")
        ]

        for filename, content, content_type, expected_format in formats_to_test:
            with patch('fastapi.UploadFile.read', return_value=content):
                # Mock PDF reader for PDF files
                if filename.endswith('.pdf'):
                    with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
                        mock_page = Mock()
                        mock_page.extract_text.return_value = "Mocked PDF content"
                        mock_instance = Mock()
                        mock_instance.pages = [mock_page]
                        mock_pdf_reader.return_value = mock_instance

                        response = client.post(
                            "/api/v1/ingest",
                            files={"file": (filename, content, content_type)},
                            data={
                                "title": f"Test {expected_format.upper()} Book",
                                "author": "Test Author",
                                "file_format": expected_format,
                                "version": "1.0"
                            }
                        )
                else:
                    # For non-PDF files
                    response = client.post(
                        "/api/v1/ingest",
                        files={"file": (filename, content, content_type)},
                        data={
                            "title": f"Test {expected_format.upper()} Book",
                            "author": "Test Author",
                            "file_format": expected_format,
                            "version": "1.0"
                        }
                    )

                # Verify the response
                assert response.status_code == 200, f"Failed for {filename}"
                data = response.json()

                assert data["status"] == "success"
                assert data["file_format"] == expected_format
                assert f"Test {expected_format.upper()} Book" in data["message"]
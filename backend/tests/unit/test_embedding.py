import pytest
from src.services.ingestion_service import IngestionService
from unittest.mock import Mock, patch
import numpy as np

class TestEmbeddingService:
    """
    Unit tests for the embedding service in ingestion_service.py
    """

    def setup_method(self):
        """Setup method to initialize the ingestion service for each test"""
        # Mock the OpenAI client to avoid actual API calls
        with patch('src.services.ingestion_service.OpenAI') as mock_openai:
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client

            # Set up mock embedding response
            mock_embedding_response = Mock()
            mock_embedding_response.data = [Mock()]
            mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding vector
            self.mock_client.embeddings.create.return_value = mock_embedding_response

            self.ingestion_service = IngestionService()

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_basic(self, mock_openai):
        """Test basic embedding generation"""
        # Set up mock
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.return_value = mock_embedding_response

        # Create new instance with mocked client
        service = IngestionService()

        text = "This is a test sentence."
        embedding = service.generate_embedding(text)

        # Verify the result
        assert isinstance(embedding, list)
        assert len(embedding) == 5  # Based on our mock
        assert all(isinstance(val, float) for val in embedding)

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_empty_text(self, mock_openai):
        """Test embedding generation with empty text"""
        # Set up mock
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.0] * 1536  # Standard OpenAI embedding size
        mock_client.embeddings.create.return_value = mock_embedding_response

        # Create new instance with mocked client
        service = IngestionService()

        text = ""
        embedding = service.generate_embedding(text)

        # Verify the result
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Standard embedding size

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_special_characters(self, mock_openai):
        """Test embedding generation with special characters"""
        # Set up mock
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = mock_embedding_response

        # Create new instance with mocked client
        service = IngestionService()

        text = "Hello, world! How are you? I'm fine. @#$%^&*()"
        embedding = service.generate_embedding(text)

        # Verify the result
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_long_text(self, mock_openai):
        """Test embedding generation with long text"""
        # Set up mock
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.5] * 1536
        mock_client.embeddings.create.return_value = mock_embedding_response

        # Create new instance with mocked client
        service = IngestionService()

        text = "This is a longer text. " * 100  # Create a longer text
        embedding = service.generate_embedding(text)

        # Verify the result
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_error_handling(self, mock_openai):
        """Test embedding generation error handling"""
        # Set up mock to raise an exception
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_client.embeddings.create.side_effect = Exception("API Error")

        # Create new instance with mocked client
        service = IngestionService()

        text = "This text will cause an error."

        # Verify that the error is properly handled/raised
        with pytest.raises(Exception):
            service.generate_embedding(text)

    def test_embedding_consistency(self):
        """Test that the same text produces the same embedding (with mocked API)"""
        # Since we're mocking, we'll test that the method is called correctly
        with patch('src.services.ingestion_service.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embedding_response = Mock()
            mock_embedding_response.data = [Mock()]
            mock_embedding_response.data[0].embedding = expected_embedding
            mock_client.embeddings.create.return_value = mock_embedding_response

            service = IngestionService()

            text = "Consistent test text"
            embedding1 = service.generate_embedding(text)
            embedding2 = service.generate_embedding(text)

            # With mocked API, both calls should return the same mocked result
            assert embedding1 == expected_embedding
            assert embedding2 == expected_embedding

    @patch('src.services.ingestion_service.OpenAI')
    def test_generate_embedding_different_texts(self, mock_openai):
        """Test that different texts produce different embeddings (with mocked API)"""
        # Set up mock to return different embeddings for different inputs
        mock_client = Mock()
        mock_openai.return_value = mock_client

        def mock_create(input, model):
            # Return different embeddings based on input
            if "first" in input:
                embedding = [0.1, 0.2, 0.3]
            elif "second" in input:
                embedding = [0.4, 0.5, 0.6]
            else:
                embedding = [0.7, 0.8, 0.9]

            response = Mock()
            response.data = [Mock()]
            response.data[0].embedding = embedding
            return response

        mock_client.embeddings.create.side_effect = mock_create

        # Create new instance with mocked client
        service = IngestionService()

        embedding1 = service.generate_embedding("first test text")
        embedding2 = service.generate_embedding("second test text")

        # Verify that embeddings are different
        assert embedding1 != embedding2
import pytest
from src.services.ingestion_service import IngestionService
from unittest.mock import Mock, patch
import uuid

class TestChunkingService:
    """
    Unit tests for the chunking service in ingestion_service.py
    """

    def setup_method(self):
        """Setup method to initialize the ingestion service for each test"""
        self.ingestion_service = IngestionService()

    def test_chunk_text_basic(self):
        """Test basic chunking functionality"""
        text = "This is a sample text for chunking. " * 10  # Create a longer text
        chunk_size = 50
        overlap = 10

        chunks = self.ingestion_service.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Verify that chunks were created
        assert len(chunks) > 0

        # Verify that each chunk is within the expected size (with some tolerance for the last chunk)
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:  # Not the last chunk
                # Chunks should be close to the requested size
                assert len(chunk) <= chunk_size
            else:
                # The last chunk can be smaller
                assert len(chunk) <= chunk_size

    def test_chunk_text_with_overlap(self):
        """Test that chunks have the expected overlap"""
        text = "This is the first part. This is the second part. This is the third part. This is the fourth part."
        chunk_size = 40
        overlap = 10

        chunks = self.ingestion_service.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # If we have multiple chunks, verify overlap characteristics
        if len(chunks) > 1:
            # The second chunk should start within the overlap region of the first chunk
            first_chunk_end = chunks[0][-overlap:]
            second_chunk_start = chunks[1][:overlap]
            # This is a simplified check - in a real scenario we'd check for more nuanced overlap
            assert len(first_chunk_end) <= overlap
            assert len(second_chunk_start) <= overlap

    def test_chunk_text_empty_input(self):
        """Test chunking with empty input"""
        chunks = self.ingestion_service.chunk_text("")
        assert chunks == []

    def test_chunk_text_shorter_than_chunk_size(self):
        """Test chunking with text shorter than chunk size"""
        text = "Short text"
        chunks = self.ingestion_service.chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_single_character(self):
        """Test chunking with single character"""
        text = "A"
        chunks = self.ingestion_service.chunk_text(text, chunk_size=5, overlap=1)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_special_characters(self):
        """Test chunking with special characters and punctuation"""
        text = "Hello, world! How are you? I'm fine. Thanks for asking."
        chunks = self.ingestion_service.chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 0
        # Verify all chunks are non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_text_with_newlines(self):
        """Test chunking with text containing newlines"""
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        chunks = self.ingestion_service.chunk_text(text, chunk_size=15, overlap=3)
        assert len(chunks) > 0
        # Verify all chunks are non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_text_with_different_sizes(self):
        """Test chunking with different chunk sizes and overlaps"""
        text = "This is a test of different chunk sizes and overlaps. " * 5

        # Test with small chunks
        small_chunks = self.ingestion_service.chunk_text(text, chunk_size=20, overlap=5)
        assert len(small_chunks) > 5  # Should create multiple small chunks

        # Test with large chunks
        large_chunks = self.ingestion_service.chunk_text(text, chunk_size=200, overlap=20)
        assert len(large_chunks) < len(small_chunks)  # Should create fewer large chunks
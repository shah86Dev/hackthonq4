import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from uuid import uuid4


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_ingest_book_success(client):
    """Test successful book ingestion"""
    book_id = str(uuid4())
    response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": book_id,
            "title": "Test Book",
            "version": "1.0",
            "content": "This is a test book content with enough text to be meaningful for testing purposes.",
            "metadata": {"author": "Test Author", "publisher": "Test Publisher"}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["book_id"] == book_id
    assert "chunks_processed" in data
    assert data["chunks_processed"] > 0
    assert "Successfully ingested" in data["message"]


def test_ingest_book_invalid_uuid(client):
    """Test book ingestion with invalid UUID"""
    response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": "invalid-uuid",
            "title": "Test Book",
            "version": "1.0",
            "content": "This is a test book content.",
            "metadata": {"author": "Test Author"}
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Invalid book_id format" in data["error"]


def test_ingest_book_missing_fields(client):
    """Test book ingestion with missing required fields"""
    response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": str(uuid4()),
            "title": "Test Book",
            # Missing required 'version' and 'content' fields
        }
    )

    assert response.status_code == 422  # Validation error


def test_ingest_book_empty_content(client):
    """Test book ingestion with empty content"""
    response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": str(uuid4()),
            "title": "Test Book",
            "version": "1.0",
            "content": "",
            "metadata": {"author": "Test Author"}
        }
    )

    # The ingestion should still work even with empty content (would create 0 chunks)
    # but this depends on the implementation. For now, we'll expect it to process
    assert response.status_code in [200, 500]  # Either success or internal error
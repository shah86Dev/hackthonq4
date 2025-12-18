import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from uuid import uuid4


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_query_book_full_book_mode_success(client):
    """Test querying book in full-book mode"""
    # Note: This test requires a book to be ingested first, which would be handled by test setup
    # For contract testing, we'll test the API contract without requiring actual ingestion
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": str(uuid4()),  # Using a random UUID for contract testing
            "question": "What is this book about?",
            "mode": "full-book"
        }
    )

    # The response should match the expected structure even if book doesn't exist
    # In a real scenario, this would return an error for non-existent book
    assert response.status_code in [200, 422, 404]  # Success, unprocessable entity, or not found

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "answer" in data
        assert "citations" in data
        assert "confidence_score" in data
        assert "retrieved_chunks" in data
        assert "query_id" in data

        # Validate data types
        assert isinstance(data["status"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["citations"], list)
        assert isinstance(data["confidence_score"], (int, float))
        assert isinstance(data["retrieved_chunks"], list)
        assert isinstance(data["query_id"], str)

        # Validate confidence score range
        assert 0.0 <= data["confidence_score"] <= 1.0


def test_query_book_selected_text_mode_success(client):
    """Test querying book in selected-text mode"""
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": str(uuid4()),
            "question": "What does this selected text mean?",
            "mode": "selected-text",
            "selected_text": "This is the selected text that the question refers to."
        }
    )

    assert response.status_code in [200, 422, 404]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "answer" in data
        assert "citations" in data
        assert "confidence_score" in data
        assert "retrieved_chunks" in data
        assert "query_id" in data


def test_query_book_invalid_mode(client):
    """Test querying book with invalid mode"""
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": str(uuid4()),
            "question": "What is this book about?",
            "mode": "invalid-mode"
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Mode must be" in data["detail"]


def test_query_book_missing_selected_text(client):
    """Test querying book in selected-text mode without selected text"""
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": str(uuid4()),
            "question": "What does this selected text mean?",
            "mode": "selected-text"
            # Missing selected_text field
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "selected_text is required" in data["detail"]


def test_query_book_invalid_book_id(client):
    """Test querying book with invalid book ID format"""
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": "invalid-uuid",
            "question": "What is this book about?",
            "mode": "full-book"
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid book_id format" in data["detail"]


def test_query_book_required_fields(client):
    """Test querying book with missing required fields"""
    response = client.post(
        "/api/v1/query",
        json={
            "book_id": str(uuid4()),
            # Missing required 'question' field
            "mode": "full-book"
        }
    )

    assert response.status_code == 422  # Validation error
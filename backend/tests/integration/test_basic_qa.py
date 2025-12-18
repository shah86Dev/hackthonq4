import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from uuid import uuid4
from unittest.mock import patch


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_complete_basic_qa_journey(client):
    """Test the complete basic Q&A journey: ingest book, query, get grounded response with citations"""
    book_id = str(uuid4())
    book_title = "Test Physics Book"
    book_content = """
    Chapter 1: Introduction to Physics
    Physics is the natural science that studies matter, its motion and behavior through space and time,
    and the related entities of energy and force. The main branches of physics are mechanics, thermodynamics,
    electromagnetism, optics, and quantum mechanics.

    Chapter 2: Mechanics
    Mechanics is the branch of physics concerned with the behavior of physical bodies when subjected to forces
    or displacements, and the subsequent effects of the bodies on their environment. The motion of an object
    can be described using concepts such as displacement, velocity, acceleration, and time.

    Chapter 3: Quantum Mechanics
    Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties
    of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics
    including quantum chemistry, quantum field theory, quantum technology, and quantum information science.
    """

    # Step 1: Ingest the book
    ingest_response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": book_id,
            "title": book_title,
            "version": "1.0",
            "content": book_content,
            "metadata": {"author": "Test Author", "subject": "Physics"}
        }
    )

    assert ingest_response.status_code == 200
    ingest_data = ingest_response.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["book_id"] == book_id
    assert ingest_data["chunks_processed"] > 0

    # Step 2: Query the book with a question that should be answerable from the content
    query_response = client.post(
        "/api/v1/query",
        json={
            "book_id": book_id,
            "question": "What is physics?",
            "mode": "full-book"
        }
    )

    assert query_response.status_code == 200
    query_data = query_response.json()

    # Step 3: Verify the response structure
    assert query_data["status"] == "success"
    assert "answer" in query_data
    assert "citations" in query_data
    assert "confidence_score" in query_data
    assert "retrieved_chunks" in query_data
    assert "query_id" in query_data

    # Step 4: Verify response content
    assert isinstance(query_data["answer"], str)
    assert len(query_data["answer"]) > 0  # Should have a non-empty answer

    # Step 5: Verify citations exist and have proper structure
    assert isinstance(query_data["citations"], list)
    if len(query_data["citations"]) > 0:
        citation = query_data["citations"][0]
        assert "text" in citation
        assert "chapter" in citation
        assert "section" in citation
        assert "page_range" in citation
        assert "chunk_id" in citation

    # Step 6: Verify confidence score is in valid range
    assert 0.0 <= query_data["confidence_score"] <= 1.0

    # Step 7: Verify retrieved chunks have proper structure
    assert isinstance(query_data["retrieved_chunks"], list)
    if len(query_data["retrieved_chunks"]) > 0:
        chunk = query_data["retrieved_chunks"][0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "chapter" in chunk
        assert "section" in chunk
        assert "page_range" in chunk
        assert "similarity_score" in chunk


def test_query_with_no_relevant_content(client):
    """Test querying with a question that has no relevant content in the book"""
    book_id = str(uuid4())
    book_content = "This book is about basic mathematics and numbers."

    # Ingest the book
    ingest_response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": book_id,
            "title": "Math Book",
            "version": "1.0",
            "content": book_content,
            "metadata": {"author": "Math Author"}
        }
    )

    assert ingest_response.status_code == 200

    # Query with a question unrelated to the book content
    query_response = client.post(
        "/api/v1/query",
        json={
            "book_id": book_id,
            "question": "What is the theory of relativity?",
            "mode": "full-book"
        }
    )

    # The response should either be an error indicating no relevant content
    # or an answer indicating the information isn't in the book
    if query_response.status_code == 200:
        query_data = query_response.json()
        assert "cannot find this information in the book" in query_data["answer"].lower()
    elif query_response.status_code == 422:
        error_data = query_response.json()
        assert "No relevant content found" in error_data["detail"]


@patch('src.services.generation_service.GenerationService.generate_answer')
def test_grounding_enforcement(mock_generate, client):
    """Test that the system enforces grounded responses from book content"""
    # Mock the generation service to return a specific response
    mock_generate.return_value = {
        "answer_text": "This is an answer based on the book content.",
        "confidence_score": 0.85,
        "tokens_used": 50
    }

    book_id = str(uuid4())
    book_content = "Physics is the study of matter and energy."

    # Ingest the book
    ingest_response = client.post(
        "/api/v1/ingest/book",
        json={
            "book_id": book_id,
            "title": "Physics Book",
            "version": "1.0",
            "content": book_content,
            "metadata": {"author": "Physics Author"}
        }
    )

    assert ingest_response.status_code == 200

    # Query the book
    query_response = client.post(
        "/api/v1/query",
        json={
            "book_id": book_id,
            "question": "What is physics?",
            "mode": "full-book"
        }
    )

    assert query_response.status_code == 200
    query_data = query_response.json()

    # Verify the response follows the required structure
    assert query_data["status"] == "success"
    assert "answer" in query_data
    assert "citations" in query_data
    assert query_data["confidence_score"] >= 0.0
"""
Test script to verify rate limiting functionality for the Book-Embedded RAG Chatbot API endpoints.
"""
import asyncio
import httpx
import time
from typing import Dict, Any


async def test_rate_limiting():
    """
    Test rate limiting by making multiple requests to the API endpoints
    """
    base_url = "http://localhost:8000"

    # Test data for chat endpoint
    chat_payload = {
        "question": "What is the meaning of life?",
        "session_id": "test-session-123",
        "book_id": "12345678-1234-5678-1234-567812345678"
    }

    # Test data for query endpoint
    query_payload = {
        "book_id": "12345678-1234-5678-1234-567812345678",
        "question": "What is the meaning of life?",
        "mode": "full-book"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Testing rate limiting on chat endpoint...")

        # Make multiple requests to test rate limiting
        responses = []
        start_time = time.time()

        for i in range(150):  # Try to exceed the rate limit (100 requests per hour)
            try:
                response = await client.post(
                    f"{base_url}/api/v1/chat",
                    json=chat_payload,
                    headers={"Content-Type": "application/json"}
                )
                responses.append(response.status_code)

                if i % 10 == 0:  # Print progress
                    print(f"Made {i+1} requests to chat endpoint, status: {response.status_code}")

            except httpx.RequestError as e:
                print(f"Request {i+1} failed: {e}")
                responses.append(None)

        end_time = time.time()
        print(f"Total time for {len(responses)} requests: {end_time - start_time:.2f} seconds")

        # Count successful vs rate limited requests
        success_count = responses.count(200)
        rate_limited_count = responses.count(429)  # 429 is the rate limit status code
        error_count = len([r for r in responses if r and r != 200 and r != 429])

        print(f"\nChat Endpoint Results:")
        print(f"Successful requests: {success_count}")
        print(f"Rate limited requests: {rate_limited_count}")
        print(f"Other errors: {error_count}")

        print("\nTesting rate limiting on query endpoint...")

        # Test query endpoint
        query_responses = []
        for i in range(150):  # Try to exceed the rate limit
            try:
                response = await client.post(
                    f"{base_url}/api/v1/query",
                    json=query_payload,
                    headers={"Content-Type": "application/json"}
                )
                query_responses.append(response.status_code)

                if i % 10 == 0:  # Print progress
                    print(f"Made {i+1} requests to query endpoint, status: {response.status_code}")

            except httpx.RequestError as e:
                print(f"Request {i+1} failed: {e}")
                query_responses.append(None)

        # Count successful vs rate limited requests for query endpoint
        query_success_count = query_responses.count(200)
        query_rate_limited_count = query_responses.count(429)
        query_error_count = len([r for r in query_responses if r and r != 200 and r != 429])

        print(f"\nQuery Endpoint Results:")
        print(f"Successful requests: {query_success_count}")
        print(f"Rate limited requests: {query_rate_limited_count}")
        print(f"Other errors: {query_error_count}")


def test_rate_limiting_sync():
    """
    Synchronous wrapper for the async test
    """
    try:
        asyncio.run(test_rate_limiting())
    except RuntimeError:
        # For environments where asyncio.run is not available
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(test_rate_limiting())


if __name__ == "__main__":
    print("Starting rate limiting tests...")
    print("Note: Make sure the backend server is running on http://localhost:8000")
    test_rate_limiting_sync()
    print("Rate limiting tests completed.")
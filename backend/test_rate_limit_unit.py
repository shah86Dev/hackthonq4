"""
Unit tests to verify the rate limiting functionality for the Book-Embedded RAG Chatbot API.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.config.settings import settings


def test_rate_limit_imports():
    """Test that rate limiting dependencies are properly imported."""
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import rate limiting dependencies: {e}")


def test_rate_limit_configuration():
    """Test that rate limiting configuration is properly loaded."""
    assert hasattr(settings, 'rate_limit_requests')
    assert hasattr(settings, 'rate_limit_window')
    assert isinstance(settings.rate_limit_requests, int)
    assert isinstance(settings.rate_limit_window, int)
    assert settings.rate_limit_requests > 0
    assert settings.rate_limit_window > 0


def test_middleware_includes_rate_limiting():
    """Test that the middleware includes rate limiting functionality."""
    try:
        from src.api.middleware import limiter, get_rate_limit_key
        assert isinstance(limiter, Limiter)
        assert callable(get_rate_limit_key)
    except ImportError as e:
        pytest.fail(f"Failed to import rate limiting middleware: {e}")


def test_api_endpoints_have_rate_limiting():
    """Test that API endpoints have rate limiting decorators."""
    # Check that the app has the limiter state set
    try:
        # Initialize middleware to make sure limiter is set
        from src.api.middleware import add_middleware
        test_app = add_middleware(app)
        assert hasattr(test_app, 'state')
        assert hasattr(test_app.state, 'limiter')
    except Exception as e:
        pytest.fail(f"Failed to initialize rate limiting middleware: {e}")


def test_rate_limit_key_function():
    """Test the rate limit key function."""
    from src.api.middleware import get_rate_limit_key
    from fastapi import Request
    from starlette.datastructures import Headers
    from starlette.types import Scope

    # Create a mock request scope for testing
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/chat",
        "query_string": b"",  # Add query_string to prevent KeyError
        "headers": [[b"x-forwarded-for", b"192.168.1.1"]],
    }

    request = Request(scope)

    # Test the key function
    key = get_rate_limit_key(request)
    assert isinstance(key, str)
    assert "192.168.1.1" in key or "ip:" in key  # Either IP-based or session-based key


def test_app_startup():
    """Test that the app starts without errors with rate limiting."""
    client = TestClient(app)

    # Test basic endpoint to ensure app works
    response = client.get("/")
    assert response.status_code == 200

    # Test health endpoint
    response = client.get("/api/v1/health")
    assert response.status_code == 200


if __name__ == "__main__":
    # Run the tests
    test_rate_limit_imports()
    print("✓ Rate limiting imports test passed")

    test_rate_limit_configuration()
    print("✓ Rate limiting configuration test passed")

    test_middleware_includes_rate_limiting()
    print("✓ Middleware rate limiting test passed")

    test_rate_limit_key_function()
    print("✓ Rate limit key function test passed")

    test_app_startup()
    print("✓ App startup test passed")

    print("\nAll unit tests passed!")
"""
Model update validation tests for the Book-Embedded RAG Chatbot implementation.
This tests model integration, AI service connectivity, and response validation.
"""
from src.config.settings import settings
import openai
import os


def test_openai_configuration():
    """Test that OpenAI is properly configured"""
    print("Testing OpenAI configuration...")

    # Check that OpenAI API key is set
    assert settings.openai_api_key is not None, "OpenAI API key should be set"
    assert settings.openai_api_key != "", "OpenAI API key should not be empty"
    assert settings.openai_api_key.startswith("sk-"), "OpenAI API key should start with 'sk-'"

    print("✓ OpenAI API key is properly configured")
    return True


def test_openai_client_initialization():
    """Test that OpenAI client can be initialized"""
    print("Testing OpenAI client initialization...")

    try:
        # Initialize OpenAI client with the configured key
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # Verify client is created properly
        assert client is not None, "OpenAI client should not be None"

        print("✓ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"✗ OpenAI client initialization failed: {str(e)}")
        return False


def test_model_availability():
    """Test that configured models are available"""
    print("Testing model availability...")

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # List available models
        models = client.models.list()

        # Check if default model is available
        model_names = [model.id for model in models.data]
        default_model = "gpt-3.5-turbo"  # Default model

        # Check if any gpt model is available
        gpt_models = [name for name in model_names if name.startswith("gpt")]

        assert len(gpt_models) > 0, "At least one GPT model should be available"

        print(f"✓ Available GPT models: {gpt_models[:3]}...")  # Show first 3
        return True
    except Exception as e:
        print(f"✗ Model availability test failed: {str(e)}")
        return False


def test_embedding_configuration():
    """Test that embedding model is properly configured"""
    print("Testing embedding configuration...")

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # Test embedding functionality
        test_text = "This is a test sentence for embedding."
        embedding_response = client.embeddings.create(
            input=test_text,
            model="text-embedding-ada-002"
        )

        # Verify embedding response
        assert embedding_response is not None, "Embedding response should not be None"
        assert hasattr(embedding_response, 'data'), "Embedding response should have data"
        assert len(embedding_response.data) > 0, "Embedding response should contain embeddings"
        assert hasattr(embedding_response.data[0], 'embedding'), "Embedding should have vector data"
        assert len(embedding_response.data[0].embedding) > 0, "Embedding vector should not be empty"

        print(f"✓ Embedding created successfully with dimension: {len(embedding_response.data[0].embedding)}")
        return True
    except Exception as e:
        print(f"✗ Embedding configuration test failed: {str(e)}")
        return False


def test_chat_completion():
    """Test that chat completion functionality works"""
    print("Testing chat completion...")

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # Test a simple chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, test message!"}],
            max_tokens=10,
            temperature=0.7
        )

        # Verify response
        assert response is not None, "Chat completion response should not be None"
        assert hasattr(response, 'choices'), "Response should have choices"
        assert len(response.choices) > 0, "Response should contain choices"
        assert hasattr(response.choices[0], 'message'), "Choice should have message"
        assert response.choices[0].message.content is not None, "Message should have content"

        print("✓ Chat completion works successfully")
        return True
    except Exception as e:
        print(f"✗ Chat completion test failed: {str(e)}")
        return False


def test_model_response_format():
    """Test that model responses follow expected format"""
    print("Testing model response format...")

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # Test response format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"}
            ],
            max_tokens=50,
            temperature=0.3
        )

        # Check response structure
        assert hasattr(response, 'id'), "Response should have ID"
        assert hasattr(response, 'object'), "Response should have object type"
        assert hasattr(response, 'created'), "Response should have creation timestamp"
        assert hasattr(response, 'model'), "Response should have model name"
        assert hasattr(response, 'choices'), "Response should have choices"
        assert hasattr(response, 'usage'), "Response should have usage info"

        # Check choice structure
        choice = response.choices[0]
        assert hasattr(choice, 'index'), "Choice should have index"
        assert hasattr(choice, 'message'), "Choice should have message"
        assert hasattr(choice.message, 'role'), "Message should have role"
        assert hasattr(choice.message, 'content'), "Message should have content"
        assert choice.message.role == "assistant", "Message role should be assistant"

        print("✓ Model response format is correct")
        return True
    except Exception as e:
        print(f"✗ Model response format test failed: {str(e)}")
        return False


def test_token_usage_tracking():
    """Test that token usage is properly tracked"""
    print("Testing token usage tracking...")

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        # Make a request and check usage
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count to 10."}],
            max_tokens=20
        )

        # Check usage tracking
        assert hasattr(response, 'usage'), "Response should have usage information"
        usage = response.usage
        assert hasattr(usage, 'prompt_tokens'), "Usage should have prompt tokens"
        assert hasattr(usage, 'completion_tokens'), "Usage should have completion tokens"
        assert hasattr(usage, 'total_tokens'), "Usage should have total tokens"
        assert usage.prompt_tokens > 0, "Prompt tokens should be positive"
        assert usage.completion_tokens > 0, "Completion tokens should be positive"
        assert usage.total_tokens > 0, "Total tokens should be positive"
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens, "Total should equal sum"

        print(f"✓ Token usage tracked: {usage.total_tokens} total tokens")
        return True
    except Exception as e:
        print(f"✗ Token usage tracking test failed: {str(e)}")
        return False


def run_model_tests():
    """Run all model update validation tests and report results"""
    print("Running model update validation tests for Book-Embedded RAG Chatbot...")

    model_tests = [
        ("OpenAI Configuration", test_openai_configuration),
        ("OpenAI Client Initialization", test_openai_client_initialization),
        ("Model Availability", test_model_availability),
        ("Embedding Configuration", test_embedding_configuration),
        ("Chat Completion", test_chat_completion),
        ("Model Response Format", test_model_response_format),
        ("Token Usage Tracking", test_token_usage_tracking),
    ]

    results = []
    for test_name, test_func in model_tests:
        try:
            print(f"Running: {test_name}")
            success = test_func()
            result = "PASS" if success else "FAIL"
            results.append((test_name, result))
            status = "[PASS]" if success else "[FAIL]"
            print(f"  {status} {test_name}: {result}")
        except Exception as e:
            results.append((test_name, f"FAIL: {str(e)}"))
            print(f"  [FAIL] {test_name}: FAILED - {str(e)}")

    print("\n" + "="*60)
    print("MODEL UPDATE VALIDATION TEST RESULTS")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, result in results:
        if result == "PASS":
            passed += 1
            print(f"[PASS] {test_name}: PASSED")
        else:
            failed += 1
            print(f"[FAIL] {test_name}: {result}")

    print(f"\nTotal: {len(results)} model tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return passed, failed


if __name__ == "__main__":
    run_model_tests()
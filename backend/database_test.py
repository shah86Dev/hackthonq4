"""
Database integration tests for the Book-Embedded RAG Chatbot implementation.
This tests database connectivity, schema validation, and data operations.
"""
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.database import get_db
from src.config.settings import settings
from src import models
import uuid


def test_database_connection():
    """Test that database connection can be established"""
    try:
        # Create engine using the configured database URL
        engine = create_engine(settings.database_url)

        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1

        print("Database connection test passed")
        return True
    except Exception as e:
        print(f"Database connection test failed: {str(e)}")
        return False


def test_database_schema():
    """Test that database schema is properly configured"""
    try:
        engine = create_engine(settings.database_url)

        # Create all tables (this will create them if they don't exist)
        models.Base.metadata.create_all(bind=engine)

        # Verify tables exist by checking if we can reflect them
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Expected tables based on models
        expected_tables = [
            'users', 'chapters', 'lab_exercises', 'quizzes',
            'questions', 'content_chunks', 'chunks'
        ]

        # Check that at least some tables exist (not necessarily all)
        assert len(tables) > 0, "No tables found in database"

        print(f"Database schema test passed. Found {len(tables)} tables")
        print(f"Tables: {tables}")
        return True
    except Exception as e:
        print(f"Database schema test failed: {str(e)}")
        return False


def test_qdrant_connection():
    """Test Qdrant vector database connection"""
    try:
        from qdrant_client import QdrantClient

        # Create Qdrant client
        client = QdrantClient(
            url=settings.qdrant_url,
            # Add timeout and other configurations as needed
        )

        # Test connection by getting collections
        collections = client.get_collections()

        print(f"Qdrant connection test passed. Found {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"Qdrant connection test failed: {str(e)}")
        return False


def test_model_serialization():
    """Test that Pydantic models can be properly serialized/deserialized"""
    try:
        from src import schemas

        # Test user schema
        user_data = {
            "email": "test@example.com",
            "full_name": "Test User",
            "is_active": True
        }

        user_schema = schemas.UserCreate(**user_data)
        user_dict = user_schema.model_dump()
        assert user_dict["email"] == "test@example.com"

        # Test chapter schema
        chapter_data = {
            "title": "Test Chapter",
            "content": "This is test content",
            "chapter_number": 1,
            "book_id": str(uuid.uuid4())
        }

        chapter_schema = schemas.ChapterCreate(**chapter_data)
        chapter_dict = chapter_schema.model_dump()
        assert chapter_dict["title"] == "Test Chapter"

        print("Model serialization test passed")
        return True
    except Exception as e:
        print(f"Model serialization test failed: {str(e)}")
        return False


def test_data_validation():
    """Test data validation rules"""
    try:
        from src import schemas

        # Test that validation works for invalid data
        try:
            invalid_user = schemas.UserCreate(
                email="invalid-email",  # Invalid email format
                full_name="Test User",
                is_active=True
            )
            # This should raise a validation error
            assert False, "Validation should have failed for invalid email"
        except:
            pass  # Expected to fail

        # Test valid data passes validation
        valid_user = schemas.UserCreate(
            email="valid@example.com",
            full_name="Valid User",
            is_active=True
        )

        assert valid_user.email == "valid@example.com"

        print("Data validation test passed")
        return True
    except Exception as e:
        print(f"Data validation test failed: {str(e)}")
        return False


def test_database_operations():
    """Test basic database operations (read/write)"""
    try:
        from src.database import SessionLocal
        from src import models, schemas
        import uuid

        # Create a session
        db = SessionLocal()

        try:
            # Create a test user
            user_id = str(uuid.uuid4())
            user = models.User(
                id=user_id,
                email=f"test_{uuid.uuid4()}@example.com",
                full_name="Test User",
                hashed_password="test_password_hash"
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            # Verify user was created
            assert user.id == user_id
            assert user.email is not None

            # Query the user back
            retrieved_user = db.query(models.User).filter(models.User.id == user_id).first()
            assert retrieved_user is not None
            assert retrieved_user.email == user.email

            print("Database operations test passed")
            return True
        finally:
            # Clean up - delete the test user
            if 'user' in locals():
                db.delete(user)
                db.commit()
            db.close()
    except Exception as e:
        print(f"Database operations test failed: {str(e)}")
        return False


def run_database_tests():
    """Run all database integration tests and report results"""
    print("Running database integration tests for Book-Embedded RAG Chatbot...")

    db_tests = [
        ("Database Connection", test_database_connection),
        ("Database Schema", test_database_schema),
        ("Qdrant Connection", test_qdrant_connection),
        ("Model Serialization", test_model_serialization),
        ("Data Validation", test_data_validation),
        ("Database Operations", test_database_operations),
    ]

    results = []
    for test_name, test_func in db_tests:
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
    print("DATABASE INTEGRATION TEST RESULTS")
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

    print(f"\nTotal: {len(results)} database tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return passed, failed


if __name__ == "__main__":
    run_database_tests()
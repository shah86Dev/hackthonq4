from src.config.settings import settings

print("Testing settings import...")
print(f"Database URL: {settings.database_url}")
print(f"Qdrant URL: {settings.qdrant_url}")
print(f"OpenAI API Key: {settings.openai_api_key}")
print("Settings loaded successfully!")
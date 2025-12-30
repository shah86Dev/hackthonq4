# Environment Variables Guide

This document details all environment variables needed for local development and Vercel deployment.

## Backend Variables

### Required for Both Local and Vercel

| Variable | Description | Local Default | Vercel Required |
|----------|-------------|---------------|-----------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and generation | Set in .env | Yes |
| `QDRANT_API_KEY` | Qdrant vector database API key | Set in .env | Yes |
| `QDRANT_HOST` | Qdrant database host URL | `http://localhost:6333` (local) or your Qdrant Cloud URL | Yes |
| `DATABASE_URL` | PostgreSQL database connection string | Local database URL | Yes |
| `SECRET_KEY` | Secret key for security | `supersecretkeychangethisinproduction` | Yes |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-ada-002` | Yes |
| `GENERATION_MODEL` | OpenAI generation model | `gpt-4o` | Yes |

### Local Development Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable/disable debug mode | `True` |
| `SERVER_HOST` | Server host for local development | `0.0.0.0` |
| `SERVER_PORT` | Server port for local development | `8000` |
| `QDRANT_PORT` | Qdrant port for local development | `6333` |

### Vercel Deployment Specific

| Variable | Description | Notes |
|----------|-------------|-------|
| `DEBUG` | Enable/disable debug mode | Should be `False` in production |
| `NEON_DB_URL` | Neon Postgres connection string | Alternative to DATABASE_URL |
| `ENVIRONMENT` | Environment identifier | `production`, `staging`, or `development` |

## Frontend Variables

### Required for Both Local and Vercel

| Variable | Description | Local Default | Vercel Default |
|----------|-------------|---------------|----------------|
| `REACT_APP_BACKEND_URL` | Backend API URL | `http://localhost:8000` | Your Vercel backend URL |

## Configuration File Mapping

The backend uses `src/config/settings.py` which maps environment variables to application settings:

```python
# API Keys and connection strings
openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
neon_postgres_url: str = os.getenv("NEON_POSTGRES_URL", "")

# Application settings
debug: bool = os.getenv("DEBUG", "False").lower() == "true"
server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
server_port: int = int(os.getenv("SERVER_PORT", "8000"))

# Qdrant settings
qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")
embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Book processing settings
chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
max_book_size: int = int(os.getenv("MAX_BOOK_SIZE", "1000000"))
top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", "5"))

# Generation settings
generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o")
max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))

# Rate limiting
rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Security
secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
```

## Setting Up for Local Development

1. Create `.env` file in the `backend` directory with required variables
2. Ensure Qdrant is running locally or update QDRANT_HOST to your cloud instance
3. Ensure PostgreSQL database is accessible
4. Run the application with `uvicorn src.main:app --reload`

## Setting Up for Vercel Deployment

1. In the Vercel dashboard, go to your project settings
2. Navigate to "Environment Variables"
3. Add all required variables from the tables above
4. The `vercel.json` configuration will handle the deployment process

## Security Notes

⚠️ **Important Security Considerations:**
- Never commit actual API keys to version control
- Use different keys for development and production
- Rotate API keys regularly
- Use strong, unique secret keys
- In production, ensure DEBUG is set to False
- Use HTTPS for all production environments

## Common Issues

### Missing Environment Variables
- Error: `ValidationError` when starting the application
- Solution: Ensure all required variables are set in the environment

### Invalid API Keys
- Error: Connection refused or authentication errors
- Solution: Verify API keys are correct and have necessary permissions

### Database Connection Issues
- Error: Cannot connect to database
- Solution: Verify DATABASE_URL or NEON_DB_URL is correct and accessible
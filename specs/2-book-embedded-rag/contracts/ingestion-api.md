# API Contract: Book Ingestion

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17
**Status**: Draft

## Endpoint

`POST /ingest/book`

## Purpose

Ingest a digital book into the RAG system by parsing content, chunking text, generating embeddings, and storing in vector database.

## Request

### Headers
- `Content-Type: application/json`
- `Authorization: Bearer <token>` (if authentication required)

### Body
```json
{
  "book_id": "string (required)",
  "title": "string (required)",
  "version": "string (required)",
  "content": "string (required) - Full book content",
  "metadata": {
    "author": "string (optional)",
    "publisher": "string (optional)",
    "publication_date": "string (optional)"
  }
}
```

## Response

### Success (200 OK)
```json
{
  "status": "success",
  "book_id": "string",
  "chunks_processed": "integer",
  "message": "string"
}
```

### Error Responses

#### 400 Bad Request
```json
{
  "status": "error",
  "error": "string",
  "details": "string"
}
```

#### 401 Unauthorized
```json
{
  "status": "error",
  "error": "Authentication required"
}
```

#### 409 Conflict
```json
{
  "status": "error",
  "error": "Book already exists",
  "book_id": "string"
}
```

#### 500 Internal Server Error
```json
{
  "status": "error",
  "error": "Ingestion failed",
  "details": "string"
}
```

## Validation Rules

- `book_id` must be unique
- `title`, `version`, and `content` are required
- `content` must be at least 100 characters
- `metadata` is optional but if provided, must be valid JSON

## Business Logic

1. Validate input parameters
2. Check if book_id already exists
3. Parse book content into structured sections
4. Chunk content (300-600 tokens with 10% overlap)
5. Generate embeddings for each chunk
6. Store chunks in Qdrant with metadata
7. Store book metadata in Postgres
8. Return ingestion summary

## Security Requirements

- Validate book_id format to prevent injection
- Sanitize content to prevent malicious payloads
- Ensure proper isolation between books
- Implement rate limiting per source

## Performance Requirements

- Process 1000-page book in under 5 minutes
- Handle content up to 50MB in size
- Support concurrent ingestion requests
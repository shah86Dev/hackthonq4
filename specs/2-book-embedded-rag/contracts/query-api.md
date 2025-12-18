# API Contract: Query Processing

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17
**Status**: Draft

## Endpoints

`POST /query`
`POST /query/selected-text`

## Purpose

Process user questions against book content using either full-book RAG or selected-text-only RAG.

## Request: Full-Book Query

### Headers
- `Content-Type: application/json`
- `Authorization: Bearer <token>` (if authentication required)

### Body
```json
{
  "book_id": "string (required)",
  "question": "string (required)",
  "mode": "string (required) - 'full-book'",
  "user_id": "string (optional) - for analytics"
}
```

## Request: Selected-Text Query

### Headers
- `Content-Type: application/json`
- `Authorization: Bearer <token>` (if authentication required)

### Body
```json
{
  "book_id": "string (required)",
  "question": "string (required)",
  "selected_text": "string (required) - User-highlighted text",
  "mode": "string (required) - 'selected-text'",
  "user_id": "string (optional) - for analytics"
}
```

## Response

### Success (200 OK)
```json
{
  "status": "success",
  "answer": "string - Generated answer",
  "citations": [
    {
      "text": "string - Cited text snippet",
      "chapter": "string",
      "section": "string",
      "page_range": "string",
      "chunk_id": "string"
    }
  ],
  "confidence_score": "number - 0 to 1",
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "text": "string - Retrieved chunk text",
      "similarity_score": "number"
    }
  ],
  "query_id": "string - Unique query identifier for logging"
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

#### 404 Not Found
```json
{
  "status": "error",
  "error": "Book not found",
  "book_id": "string"
}
```

#### 422 Unprocessable Entity
```json
{
  "status": "error",
  "error": "No relevant content found in book",
  "message": "I cannot find this information in the book."
}
```

#### 429 Too Many Requests
```json
{
  "status": "error",
  "error": "Rate limit exceeded"
}
```

#### 500 Internal Server Error
```json
{
  "status": "error",
  "error": "Query processing failed",
  "details": "string"
}
```

## Validation Rules

- `book_id` must exist in the system
- `question` must be at least 5 characters
- For selected-text mode, `selected_text` is required and must be at least 10 characters
- `mode` must be either 'full-book' or 'selected-text'
- Selected-text mode must not access vector database for retrieval

## Business Logic

### Full-Book RAG:
1. Validate input parameters
2. Perform vector similarity search in Qdrant (filter by book_id)
3. Retrieve top-k relevant chunks
4. Generate answer using LLM with retrieved context
5. Extract citations from source chunks
6. Calculate confidence score
7. Log query in Postgres

### Selected-Text RAG:
1. Validate input parameters
2. Use only `selected_text` as context (no vector DB access)
3. Generate answer using LLM with selected text as context
4. Create citation from selected text metadata
5. Calculate confidence score
6. Log query in Postgres

## Security Requirements

- Validate book_id to prevent cross-book access
- Sanitize user input to prevent prompt injection
- Implement rate limiting per session/user
- Ensure selected-text mode doesn't access vector DB

## Performance Requirements

- Response time under 2 seconds (P95)
- Support concurrent queries
- Handle questions up to 1000 characters
- Handle selected text up to 10,000 characters
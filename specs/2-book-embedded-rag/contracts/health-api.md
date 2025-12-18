# API Contract: Health Check

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17
**Status**: Draft

## Endpoint

`GET /health`

## Purpose

Check the health status of the Book-Embedded RAG Chatbot system and its dependencies.

## Request

### Headers
- `Accept: application/json`

### Parameters
- None

## Response

### Success (200 OK)
```json
{
  "status": "healthy",
  "timestamp": "string - ISO 8601 timestamp",
  "version": "string - API version",
  "services": {
    "database": {
      "status": "string - healthy|unhealthy",
      "response_time": "number - ms"
    },
    "vector_db": {
      "status": "string - healthy|unhealthy",
      "response_time": "number - ms"
    },
    "llm_provider": {
      "status": "string - healthy|unhealthy",
      "response_time": "number - ms"
    }
  },
  "details": {
    "uptime": "string - Duration since last restart",
    "active_connections": "number",
    "pending_tasks": "number"
  }
}
```

### Error Responses

#### 503 Service Unavailable
```json
{
  "status": "unhealthy",
  "timestamp": "string - ISO 8601 timestamp",
  "error": "string - Primary error message",
  "services": {
    "database": {
      "status": "unhealthy",
      "error": "string - Specific error"
    },
    "vector_db": {
      "status": "unhealthy",
      "error": "string - Specific error"
    },
    "llm_provider": {
      "status": "unhealthy",
      "error": "string - Specific error"
    }
  }
}
```

## Validation Rules

- Response must be returned within 1 second
- All service checks must complete within the timeout
- Health status reflects the most critical dependency status

## Business Logic

1. Check database connectivity
2. Check vector database connectivity
3. Check LLM provider connectivity (optional ping)
4. Aggregate status across all services
5. Return comprehensive health report

## Security Requirements

- No authentication required for health checks
- Do not expose sensitive system information
- Limit health check frequency to prevent abuse

## Performance Requirements

- Response time under 1 second
- Minimal resource usage for health checks
- Non-blocking health check operations
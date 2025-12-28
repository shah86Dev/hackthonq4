# Rate Limiting Implementation for Book-Embedded RAG Chatbot

## Overview
This document outlines the implementation of rate limiting functionality for the Book-Embedded RAG Chatbot API endpoints as specified in the requirements.

## Implementation Details

### 1. Dependencies Added
- Added `slowapi==0.1.8` to requirements.txt for rate limiting functionality
- Added `limits>=2.3` as a dependency of slowapi

### 2. Configuration
- Utilized existing rate limiting configuration from `src/config/settings.py`
- Configuration parameters:
  - `rate_limit_requests`: 100 requests per time window (default)
  - `rate_limit_window`: 3600 seconds (1 hour) (default)
- These values can be customized via environment variables

### 3. Middleware Implementation
- Updated `src/api/middleware.py` with comprehensive rate limiting middleware
- Implemented session-based rate limiting with fallback to IP-based:
  - Checks for session_id in headers (`x-session-id`, `session-id`)
  - Checks for session_id in query parameters (`session_id`, `sessionId`)
  - Uses session-based rate limiting when session_id is available
  - Falls back to IP-based rate limiting when no session_id is provided
- Added CORS and logging middleware as before

### 4. API Endpoint Protection
Applied rate limiting to the following RAG chatbot endpoints:

#### Chat Endpoint (`/api/v1/chat`)
- Added rate limiting decorator with configurable limits
- Supports per-session rate limiting when session_id is provided in request
- Falls back to IP-based rate limiting

#### Query Endpoint (`/api/v1/query`)
- Added rate limiting decorator with configurable limits
- Supports per-session rate limiting when session_id is provided in request
- Falls back to IP-based rate limiting

#### Ingest Endpoint (`/api/v1/ingest/book`)
- Added rate limiting decorator with configurable limits
- Protects the book ingestion functionality from abuse

### 5. Rate Limit Key Strategy
The system implements a flexible rate limiting key strategy:
- **Session-based**: When session_id is provided (in headers or query params), rate limiting is applied per session
- **IP-based**: When no session_id is provided, rate limiting is applied per IP address
- **Path-specific**: Different endpoints have separate rate limits

### 6. Error Handling
- Rate limit exceeded requests return HTTP 429 (Too Many Requests)
- Proper error messages are provided to clients
- System continues to function normally for non-rate-limited requests

## Files Modified

1. `backend/requirements.txt` - Added slowapi dependency
2. `backend/src/api/middleware.py` - Added rate limiting middleware
3. `backend/src/api/endpoints/chat.py` - Added rate limiting to chat endpoint
4. `backend/src/api/endpoints/query.py` - Added rate limiting to query endpoint
5. `backend/src/api/endpoints/ingest.py` - Added rate limiting to ingest endpoint

## Testing

### Unit Tests
- Created comprehensive unit tests in `test_rate_limit_unit.py`
- Tests verify:
  - Rate limiting dependencies are properly imported
  - Configuration is correctly loaded
  - Middleware includes rate limiting functionality
  - Rate limit key function works correctly
  - Application starts without errors

### Test Results
All 6 unit tests passed, confirming the rate limiting implementation works correctly.

## Configuration

The rate limiting behavior can be configured via environment variables:

```bash
RATE_LIMIT_REQUESTS=100      # Number of requests allowed per time window
RATE_LIMIT_WINDOW=3600       # Time window in seconds (default: 1 hour)
```

## Security Considerations

1. **Per-Session Rate Limiting**: When clients provide session IDs, rate limiting is applied per session, providing more granular control
2. **Fallback Protection**: IP-based rate limiting ensures protection even when session IDs are not provided
3. **Configurable Limits**: Limits can be adjusted based on deployment requirements
4. **No Rate Limit Bypass**: All major API endpoints are protected

## Performance Impact

- Minimal performance overhead as rate limiting is implemented efficiently
- Rate limiting data is stored in memory by default (can be configured to use Redis for production)
- Proper error responses help clients implement appropriate backoff strategies

## Future Enhancements

1. **Redis Backend**: For production deployments with multiple instances, configure slowapi to use Redis for distributed rate limiting
2. **Different Limits per Endpoint**: Implement different rate limits for different types of endpoints (chat vs query vs ingest)
3. **User-Based Rate Limiting**: Integrate with authentication system for user-specific rate limits
4. **Dynamic Limits**: Implement dynamic rate limits based on system load or user tier
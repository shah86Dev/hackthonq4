# Data Model: Book-Embedded RAG Chatbot

## Overview
This document defines the data models for the Book-Embedded RAG Chatbot system, including entities, relationships, and validation rules.

## Entity: Book Chunk
Represents a segment of book content with embedding vector and metadata for retrieval.

### Attributes
- **id** (string, required): Unique identifier for the chunk
- **text_content** (string, required): The actual text content of the chunk (500-1000 chars)
- **book_id** (string, required): Reference to the parent book
- **page_number** (integer, optional): Page number in the original book
- **section** (string, optional): Section/chapter title
- **position** (integer, required): Sequential position in the book
- **embedding_vector** (array of floats, required): 1536-dimensional embedding vector
- **metadata** (object, optional): Additional metadata (word count, etc.)

### Validation Rules
- text_content must be between 500-1000 characters
- page_number must be positive if provided
- position must be positive
- embedding_vector must be exactly 1536 dimensions

### Relationships
- Belongs to one Book (via book_id)
- Referenced by multiple Query Logs

## Entity: Query Log
Records user interactions for analytics and system improvement.

### Attributes
- **id** (string, required): Unique identifier for the query log
- **question** (string, required): The user's original question
- **selected_text** (string, optional): Text selected by user (if any)
- **retrieved_chunks** (array of strings, required): IDs of chunks retrieved for context
- **response** (string, required): The system's response
- **book_id** (string, required): Reference to the book queried
- **user_id** (string, optional): User identifier (if available)
- **timestamp** (datetime, required): When the query was made
- **response_time_ms** (integer, required): Time taken to generate response
- **grounded_confidence** (float, optional): Confidence score for grounding in source

### Validation Rules
- question must not be empty
- response must not be empty
- retrieved_chunks must contain valid chunk IDs
- response_time_ms must be positive
- grounded_confidence must be between 0 and 1 if provided

### Relationships
- References multiple Book Chunks (via retrieved_chunks)
- Belongs to one Book (via book_id)

## Entity: Book Metadata
Contains information about processed books including format, size, and processing status.

### Attributes
- **id** (string, required): Unique identifier for the book
- **title** (string, required): Title of the book
- **author** (string, optional): Author of the book
- **format** (string, required): Format of the source (PDF, Markdown, etc.)
- **total_pages** (integer, optional): Total pages in the book
- **total_chunks** (integer, required): Number of chunks created
- **processing_status** (string, required): Status (pending, processing, completed, failed)
- **created_at** (datetime, required): When the book was added
- **processed_at** (datetime, optional): When processing was completed
- **file_size_bytes** (integer, optional): Size of the original file
- **embedding_model** (string, required): Model used for embeddings
- **chunk_config** (object, required): Configuration used for chunking (size, overlap)

### Validation Rules
- title must not be empty
- total_pages must be positive if provided
- total_chunks must be positive
- processing_status must be one of the allowed values
- format must be one of the supported formats (PDF, Markdown, etc.)
- chunk_config must include size and overlap values

### Relationships
- Has many Book Chunks (one-to-many relationship)
- Has many Query Logs (one-to-many relationship)

## API Data Contracts

### Chat Request
```json
{
  "question": "string (required)",
  "selected_text": "string (optional)"
}
```

### Chat Response
```json
{
  "answer": "string (required)",
  "citations": "array of chunk IDs (optional)",
  "confidence": "float (optional, 0-1)"
}
```

### Ingest Request
```json
{
  "title": "string (required)",
  "author": "string (optional)",
  "content": "string (required)",
  "format": "string (required, PDF/MD)",
  "chunk_size": "integer (optional, default 1000)",
  "chunk_overlap": "integer (optional, default 200)"
}
```

### Ingest Response
```json
{
  "book_id": "string (required)",
  "status": "string (required)",
  "total_chunks": "integer (required)",
  "processing_time_ms": "integer (required)"
}
```

## State Transitions

### Book Processing States
```
pending -> processing -> completed
           |-> failed
```

- `pending`: Book added to queue for processing
- `processing`: Chunking and embedding in progress
- `completed`: All chunks created and stored
- `failed`: Error occurred during processing

### Validation Summary
All entities follow the constitutional principle of Data Integrity and Traceability by maintaining complete traceability from source content to generated answers. Each response can be traced back to specific book chunks that provided the context.
# Data Model: Book-Embedded RAG Chatbot

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17
**Input**: Feature specification and technical plan

## Entity Relationships

### Book
- **Purpose**: Represents a digital book that has been processed for RAG functionality
- **Attributes**:
  - id (UUID/string): Unique identifier for the book
  - title (string): Title of the book
  - version (string): Version identifier for the book content
  - created_at (timestamp): When the book was ingested
  - metadata (JSON): Additional book metadata (author, publisher, etc.)

### Chunk
- **Purpose**: Represents a processed segment of book content for vector retrieval
- **Attributes**:
  - id (UUID/string): Unique identifier for the chunk
  - book_id (UUID/string): Reference to the parent book
  - chapter (string): Chapter identifier/name
  - section (string): Section identifier/name
  - page_range (string): Page range (e.g., "15-18")
  - text (string): The actual text content of the chunk
  - embedding (vector): Embedding vector for semantic search
  - chunk_id (string): Sequential identifier within the book

### Query
- **Purpose**: Represents a user question and the context of the query
- **Attributes**:
  - id (UUID/string): Unique identifier for the query
  - book_id (UUID/string): Reference to the book being queried
  - mode (enum): Retrieval mode ('full-book' or 'selected-text')
  - question (string): The user's question text
  - selected_text (string, nullable): User-highlighted text (for selected-text mode)
  - retrieved_chunk_ids (array): List of chunk IDs that were retrieved
  - latency (float): Query processing time in seconds
  - tokens_used (integer): Number of tokens consumed in the query
  - created_at (timestamp): When the query was made

### Answer
- **Purpose**: Represents the generated response to a user query
- **Attributes**:
  - id (UUID/string): Unique identifier for the answer
  - query_id (UUID/string): Reference to the original query
  - answer_text (string): The generated answer text
  - citations (JSON): Citation information with page/chapter references
  - confidence_score (float): Confidence score (0-1) of the answer
  - created_at (timestamp): When the answer was generated

### Feedback
- **Purpose**: Stores user feedback on answers for continuous improvement
- **Attributes**:
  - id (UUID/string): Unique identifier for the feedback
  - answer_id (UUID/string): Reference to the answer being rated
  - user_rating (enum): User rating ('positive', 'negative', 'neutral')
  - correction (string, nullable): User-provided correction if answer was incorrect
  - created_at (timestamp): When the feedback was submitted

## Database Schema

### Postgres Schema
```sql
-- Books table
CREATE TABLE books (
    id UUID PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Chunks table
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    chapter VARCHAR(200),
    section VARCHAR(200),
    page_range VARCHAR(50),
    text TEXT NOT NULL,
    embedding VECTOR(1536), -- Assuming OpenAI embeddings
    chunk_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Queries table
CREATE TABLE queries (
    id UUID PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    mode VARCHAR(20) NOT NULL, -- 'full-book' or 'selected-text'
    question TEXT NOT NULL,
    selected_text TEXT,
    retrieved_chunk_ids UUID[],
    latency FLOAT,
    tokens_used INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Answers table
CREATE TABLE answers (
    id UUID PRIMARY KEY,
    query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL,
    citations JSONB NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    answer_id UUID REFERENCES answers(id) ON DELETE CASCADE,
    user_rating VARCHAR(10) CHECK (user_rating IN ('positive', 'negative', 'neutral')),
    correction TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Vector Database Schema (Qdrant)

### Collection: book_chunks
- **Purpose**: Store text chunks with embeddings for semantic search
- **Vectors**: OpenAI text-embedding vectors (1536 dimensions)
- **Payload**:
  - book_id: UUID of the source book
  - chunk_id: Sequential identifier
  - chapter: Chapter name/number
  - section: Section name/number
  - page_range: Page range in the book
  - text: Original text content
  - metadata: Additional metadata as needed

## Relationships

1. **Book → Chunks**: One-to-Many (One book has many chunks)
2. **Book → Queries**: One-to-Many (One book can have many queries)
3. **Query → Answer**: One-to-One (One query generates one answer)
4. **Answer → Feedback**: One-to-Many (One answer can have multiple feedback entries)

## Constraints

1. **Referential Integrity**: Foreign key constraints ensure data consistency
2. **Book Isolation**: Queries can only access chunks from the same book_id
3. **Selected Text Mode**: When mode is 'selected-text', retrieved_chunk_ids should be empty
4. **Citation Requirements**: Every answer must have associated citations
5. **Confidence Scoring**: Confidence score must be between 0 and 1
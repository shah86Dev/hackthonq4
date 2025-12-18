# Research Notes: Book-Embedded RAG Chatbot

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17
**Status**: Completed

## Technical Decisions

### 1. Architecture Pattern: Retrieval-Augmented Generation (RAG)

**Decision**: Implement a RAG architecture with separate ingestion, retrieval, and generation components.

**Rationale**:
- RAG allows for grounded responses based on specific documents
- Separation of concerns improves maintainability
- Enables book-specific knowledge without model retraining
- Supports real-time content updates through ingestion pipeline

**Alternatives Considered**:
- Fine-tuning a model on book content: Higher cost, less flexible, harder to update
- Simple semantic search: No generative responses, limited user experience

### 2. Vector Database Selection: Qdrant Cloud

**Decision**: Use Qdrant Cloud as the vector database for storing document embeddings.

**Rationale**:
- High-performance similarity search
- Cloud-hosted option with free tier available
- Good Python SDK support
- Supports metadata filtering (important for book isolation)
- Supports hybrid search if needed in the future

**Alternatives Considered**:
- Pinecone: Good alternative but higher cost structure
- Weaviate: Feature-rich but potentially more complex setup
- Chroma: Good for prototyping but less suitable for production

### 3. Backend Framework: FastAPI

**Decision**: Use FastAPI for the backend API.

**Rationale**:
- Automatic API documentation generation
- Built-in validation and serialization
- Asynchronous support for better performance
- Strong typing support with Pydantic
- Good integration with OpenAI libraries

### 4. Embedding Model: OpenAI Text Embeddings

**Decision**: Use OpenAI's text-embedding models for generating document embeddings.

**Rationale**:
- High-quality embeddings with good semantic understanding
- Consistent performance across domains
- Easy integration with OpenAI's LLMs for generation
- Well-documented and reliable API

**Alternatives Considered**:
- Sentence Transformers: Free but requires self-hosting and maintenance
- Cohere embeddings: Good alternative but less integration with OpenAI LLMs

### 5. Frontend Embedding Approach

**Decision**: Create a lightweight JavaScript widget that can be embedded in digital books.

**Rationale**:
- Works in static book environments (Docusaurus, HTML, EPUB WebView)
- Minimal impact on book performance
- Can be loaded asynchronously
- Supports both full-book and selected-text modes

## Implementation Considerations

### 1. Chunking Strategy

**Approach**:
- Chunk size: 300-600 tokens with 10% overlap
- Split on semantic boundaries (paragraphs, sections)
- Preserve context across chunks
- Include metadata (chapter, section, page) with each chunk

**Rationale**:
- Balances retrieval precision with context availability
- Overlap helps maintain context across boundaries
- Metadata enables proper citations

### 2. Retrieval Modes

**Full-Book RAG**:
- Vector search across all book chunks
- Metadata filtering by book_id
- Top-k retrieval with configurable k

**Selected-Text RAG**:
- Bypass vector database
- Use only user-provided text as context
- Ensures strict adherence to selected text

### 3. Citation Generation

**Approach**:
- Track source chunks for each answer
- Generate citations based on chunk metadata
- Include chapter, section, and page references
- Format citations appropriately for book context

### 4. Confidence Scoring

**Approach**:
- Use embedding similarity scores as base confidence
- Apply additional heuristics (query-chunk relevance, answer certainty)
- Set threshold for when to refuse answering
- Provide confidence score with each response

## Security Considerations

### 1. Book Isolation
- Implement strict book_id filtering
- Validate book_id in all queries
- Use parameterized queries to prevent injection

### 2. Rate Limiting
- Implement per-session rate limiting
- Consider user-based limits for authenticated sessions
- Monitor for abuse patterns

### 3. Input Sanitization
- Sanitize user queries and selected text
- Prevent prompt injection attacks
- Validate all input parameters

## Performance Considerations

### 1. Latency Optimization
- Cache frequently accessed embeddings
- Optimize vector search with indexing
- Implement connection pooling for database
- Use CDN for frontend assets

### 2. Scalability
- Design for 10k+ page books
- Support concurrent users
- Implement proper resource management
- Consider async processing for ingestion

## Observability Requirements

### 1. Logging Strategy
- Log all queries with metadata
- Track retrieval performance metrics
- Monitor LLM usage and costs
- Record confidence scores and answer quality

### 2. Monitoring Metrics
- Query latency (p50, p95, p99)
- Retrieval success rate
- LLM token usage
- Error rates by type

## Potential Challenges

### 1. Large Book Handling
- Memory management during ingestion
- Vector database performance with large collections
- Chunk retrieval efficiency

### 2. Quality Assurance
- Preventing hallucinations
- Ensuring citation accuracy
- Handling ambiguous questions
- Managing low-confidence scenarios

### 3. Frontend Integration
- Compatibility with various book formats
- Cross-origin restrictions
- Performance impact on book loading
- Mobile device support

## Risk Mitigation

### 1. Provider Dependency
- Abstract LLM/vector/embedding providers behind interfaces
- Implement fallback mechanisms
- Design for easy provider switching

### 2. Cost Management
- Implement token usage tracking
- Set query limits per user/book
- Optimize embedding and generation usage

### 3. Data Privacy
- Ensure book content isolation
- Implement proper access controls
- Clear data retention policies
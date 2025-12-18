# Quickstart Guide: Book-Embedded RAG Chatbot

**Feature**: 2-book-embedded-rag
**Date**: 2025-12-17

## Overview

This guide provides the essential steps to set up and use the Book-Embedded RAG Chatbot system. The system enables digital books to have an embedded chatbot that answers questions strictly from the book's content with proper citations.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend development)
- Qdrant Cloud account (free tier available)
- Neon Serverless Postgres account
- OpenAI API key
- Digital book content in text format

## Setup Instructions

### 1. Environment Configuration

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Qdrant, Neon, and OpenAI credentials
```

### 2. Database Setup

```bash
# Set up Neon Postgres database
# Create tables using the schema defined in data-model.md

# Initialize Qdrant collection
# Create 'book_chunks' collection with 1536-dimension vectors
```

### 3. Backend Service

```bash
# Navigate to backend directory
cd backend

# Run the FastAPI application
uvicorn src.api.main:app --reload --port 8000
```

### 4. Frontend Integration

```bash
# Navigate to chatbot directory
cd chatbot

# Install frontend dependencies
npm install

# Build the embeddable widget
npm run build
```

## Usage Examples

### 1. Ingesting a Book

```bash
# Send a POST request to ingest a book
curl -X POST "http://localhost:8000/ingest/book" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Physical AI Textbook",
    "version": "1.0",
    "content": "Full text content of the book...",
    "book_id": "unique-book-id"
  }'
```

### 2. Querying with Full-Book RAG

```bash
# Ask a question using full-book RAG mode
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "unique-book-id",
    "question": "What is embodied cognition?",
    "mode": "full-book"
  }'
```

### 3. Querying with Selected-Text RAG

```bash
# Ask a question using selected-text RAG mode
curl -X POST "http://localhost:8000/query/selected-text" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "unique-book-id",
    "question": "What does this section say about neural networks?",
    "selected_text": "The section about neural networks in physical AI..."
  }'
```

## Frontend Embedding

### 1. Include the Widget

```html
<!-- Add to your digital book HTML -->
<script src="/path/to/book-embedded-rag.js"></script>

<!-- Initialize the chatbot -->
<div id="book-chatbot"></div>
<script>
  BookRAG.init({
    containerId: 'book-chatbot',
    backendUrl: 'http://localhost:8000',
    bookId: 'unique-book-id'
  });
</script>
```

### 2. Using the JavaScript API

```javascript
// Ask a question with selected text
const response = await BookRAG.askQuestion({
  question: "What is the main concept here?",
  selectedText: "Highlighted text from the book..."
});

// Render the answer with citations
BookRAG.renderAnswer(response.answer, response.citations);
```

## API Endpoints

### Ingestion
- `POST /ingest/book` - Ingest a new book into the system

### Query
- `POST /query` - Query using full-book RAG mode
- `POST /query/selected-text` - Query using selected-text RAG mode

### Health
- `GET /health` - Check system health status

## Configuration

### Environment Variables

```bash
# Backend configuration
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=your_neon_postgres_connection_string
```

## Troubleshooting

### Common Issues

1. **Embedding Generation Fails**: Ensure your OpenAI API key is valid and has sufficient quota
2. **Vector Search Returns No Results**: Verify the book was properly ingested and chunks were created
3. **CORS Errors**: Configure appropriate CORS settings in the backend
4. **Frontend Widget Not Loading**: Check that the built JavaScript file is properly referenced

### Debugging Queries

Enable debug logging by setting `DEBUG=true` in your environment variables.
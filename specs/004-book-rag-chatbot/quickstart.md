# Quickstart: Book-Embedded RAG Chatbot

## Overview
This guide provides a quick introduction to setting up and using the Book-Embedded RAG Chatbot system.

## Prerequisites
- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Qdrant Cloud account
- Neon Postgres account

## Local Setup

### 1. Environment Configuration
```bash
# Create .env file with required credentials
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cluster_url
NEON_POSTGRES_URL=your_neon_postgres_connection_string
```

### 2. Run with Docker Compose
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Build and start services
docker-compose up --build

# The API will be available at http://localhost:8000
```

### 3. Ingest a Book
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Example Book",
    "author": "Author Name",
    "content": "Full text content of the book goes here...",
    "format": "TXT"
  }'
```

### 4. Query the Chatbot
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main themes?",
    "selected_text": "Optional text selected by user"
  }'
```

## Frontend Integration

### 1. Embed the Chat Widget
Add this script to your book viewer page:

```html
<div id="book-rag-chatbot"></div>
<script src="/path/to/chat-widget.js"></script>
<script>
  BookRagChatbot.init({
    apiEndpoint: 'http://localhost:8000',
    containerId: 'book-rag-chatbot'
  });
</script>
```

### 2. Enable Text Selection
The widget automatically captures selected text using `window.getSelection()`:

```javascript
document.addEventListener('mouseup', () => {
  const selected = window.getSelection().toString();
  if (selected.trim()) {
    // Send selected text to API
    fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: 'Explain this',
        selected_text: selected
      })
    });
  }
});
```

## Production Deployment

### Kubernetes with Dapr
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f dapr/components/

# Verify deployment
kubectl get pods
kubectl port-forward service/book-rag-chatbot 8000:80
```

## Development

### Backend Development
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run backend with hot reload
cd backend
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
# Install dependencies
cd frontend
npm install

# Run frontend with hot reload
npm start
```

## Testing

### Run Unit Tests
```bash
# Backend tests
cd backend
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

### Run Contract Tests
```bash
# Validate API contracts
python -m pytest tests/contract/
```

## Architecture Overview

The system consists of:
- **Backend**: FastAPI application with agent services
- **Vector Store**: Qdrant for semantic search
- **Metadata Store**: Neon Postgres for chunk metadata
- **Frontend**: Embeddable chat widget
- **Agents**: Retrieval, Generation, and Coordinator agents

## Next Steps
1. Review the API documentation at `/docs`
2. Check the data models in `data-model.md`
3. Explore the complete task breakdown in `tasks.md` (after running `/sp.tasks`)
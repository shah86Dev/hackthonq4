# Book-Embedded RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot system designed to answer questions about book content with contextual awareness and citation support.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Features

### Core Capabilities
- **Multi-Format Book Ingestion**: Support for PDF, Markdown, and plain text formats
- **Context-Aware Question Answering**: Accurate responses grounded in book content
- **Selected Text Context**: Prioritize user-selected text when answering questions
- **Citation Support**: Answers include references to specific sections and pages
- **Scalable Processing**: Handle large books (>500 pages) efficiently
- **Distributed Processing**: Use Ray for parallel chunking and embedding

### Advanced Features
- **Real-time Text Selection**: Integrate with book viewers for contextual questioning
- **Rate Limiting**: Prevent abuse with configurable rate limits
- **Performance Monitoring**: Track response times and system metrics
- **Security Headers**: Protect against common web vulnerabilities
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Architecture

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Frontend      │────│   FastAPI        │────│  Qdrant Vector   │
│   Components    │    │   Backend        │    │  Store           │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │
                       ┌──────────────────┐
                       │   PostgreSQL     │
                       │   Metadata       │
                       └──────────────────┘
                              │
                       ┌──────────────────┐
                       │   OpenAI API     │
                       │   (Embeddings &  │
                       │   Generation)    │
                       └──────────────────┘
```

### Tech Stack
- **Backend**: FastAPI with Python 3.11+
- **Vector Store**: Qdrant Cloud for semantic search
- **Database**: PostgreSQL (Neon) for metadata storage
- **Embeddings**: OpenAI text-embedding-ada-002
- **Generation**: OpenAI GPT-4o
- **Frontend**: React components with embedding capability
- **Orchestration**: Ray for distributed processing
- **Deployment**: Docker & Kubernetes with Dapr

## Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Qdrant Cloud account
- PostgreSQL/Neon database

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/book-rag-chatbot.git
cd book-rag-chatbot
```

2. Install Python dependencies:
```bash
pip install -r backend/requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys and configuration
```

### Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-ada-002

# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content

# Database Configuration
NEON_DB_URL=postgresql://username:password@ep-xxxxxxx.us-east-1.aws.neon.tech/dbname

# Application Settings
DEBUG=False
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Chunking Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_CHUNKS=5

# Security
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Configuration

### Application Settings

The application can be configured through environment variables in the `.env` file:

- `DEBUG`: Enable/disable debug mode
- `CHUNK_SIZE`: Size of text chunks (500-1000 chars recommended)
- `CHUNK_OVERLAP`: Overlap between chunks (200 chars recommended)
- `TOP_K_CHUNKS`: Number of chunks to retrieve for context (default: 5)
- `MAX_BOOK_SIZE`: Maximum book size in characters (default: 1,000,000)
- `RATE_LIMIT_REQUESTS`: Requests allowed per window
- `RATE_LIMIT_WINDOW`: Time window in seconds for rate limiting

### Docker Configuration

Use the provided `docker-compose.yml` for local development:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - QDRANT_HOST=${QDRANT_HOST}
      - NEON_DB_URL=${NEON_DB_URL}
    depends_on:
      - qdrant
    networks:
      - rag_network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
    networks:
      - rag_network

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    networks:
      - rag_network

volumes:
  qdrant_data:

networks:
  rag_network:
    driver: bridge
```

## Usage

### Running Locally

1. Start the backend:
```bash
cd backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend (in a separate terminal):
```bash
cd frontend
npm start
```

### API Endpoints

#### Chat Endpoint
```
POST /api/v1/chat
```

Request Body:
```json
{
  "question": "What is the main theme of this book?",
  "selected_text": "Optional selected text for context",
  "book_id": "book-uuid",
  "session_id": "optional-session-id",
  "language": "en"
}
```

Response:
```json
{
  "response": "The main theme of this book is...",
  "source_chunks": [
    {
      "text": "Relevant text from the book",
      "section": "Chapter 1",
      "page_range": "1-5"
    }
  ],
  "session_id": "session-uuid",
  "confidence_score": 0.85,
  "response_time_ms": 1200,
  "tokens_used": 150
}
```

#### Ingestion Endpoint
```
POST /api/v1/ingest
```

Upload a book file (PDF, Markdown, or TXT) to process and store for RAG.

#### Health Check
```
GET /health
```

Returns system health status.

### Embedding the Chatbot

The chatbot can be embedded in any web page using the provided JavaScript snippet:

```html
<!-- Embed the chatbot -->
<div id="book-rag-chatbot"></div>

<script>
  // Load the chatbot script
  const script = document.createElement('script');
  script.src = '/path/to/chatbot-embed.js';
  script.onload = function() {
    // Initialize the chatbot
    window.BookRagEmbedder.init({
      bookId: 'your-book-id',
      apiUrl: 'http://localhost:8000',
      title: 'Book Assistant',
      theme: 'light'
    });
  };
  document.head.appendChild(script);
</script>
```

## Development

### Running Tests

Unit tests:
```bash
cd backend
python -m pytest tests/unit/
```

Integration tests:
```bash
cd backend
python -m pytest tests/integration/
```

Acceptance tests:
```bash
cd backend
python -m pytest tests/acceptance/
```

### Code Structure

```
backend/
├── src/
│   ├── api/                 # API endpoints
│   ├── models/              # Database models
│   ├── services/            # Business logic
│   ├── agents/              # AI agents
│   ├── config/              # Configuration
│   ├── utils/               # Utility functions
│   └── main.py              # Application entry point
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
└── Dockerfile               # Docker configuration

frontend/
├── src/
│   ├── components/          # React components
│   ├── services/            # Frontend services
│   └── utils/               # Utility functions
├── package.json             # Node.js dependencies
└── Dockerfile               # Docker configuration
```

## Deployment

### Kubernetes

Deploy using the provided Kubernetes manifests:

```bash
kubectl apply -f shared/k8s/deployment.yaml
kubectl apply -f shared/k8s/service.yaml
```

### Dapr Integration

The application supports Dapr for service discovery and state management:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: book-rag-statestore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis-master:6379
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify your OpenAI and Qdrant API keys are correct
2. **Rate Limiting**: Check your OpenAI usage and upgrade if necessary
3. **Memory Issues**: Large books may require increased memory allocation
4. **CORS Errors**: Ensure allowed origins are configured correctly

### Logging

The application logs to both console and file. Check the `logs/` directory for detailed logs.

### Performance

For large books (>500 pages), ensure you have sufficient resources and consider using the distributed processing capabilities with Ray.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Built with ❤️ for enhancing the reading experience with AI-powered assistance.
 

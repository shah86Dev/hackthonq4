# Quickstart: Book-Integrated RAG Chatbot

## Overview
This guide provides instructions for setting up and running the Book-Integrated RAG Chatbot.

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ (for frontend development)
- Docker and Docker Compose
- Access to OpenAI API (with text-embedding-ada-002 and GPT-4o access)
- Qdrant Cloud Free Tier account
- Neon Postgres account
- Ray cluster (for large book processing)

## Installation

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Set Up Backend Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Frontend Environment
```bash
cd frontend
npm install
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the backend directory:

```env
# OpenAI settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_GENERATION_MODEL=gpt-4o

# Qdrant settings
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content

# Neon Postgres settings
NEON_DATABASE_URL=postgresql://username:password@ep-xxxxxx.us-east-1.aws.neon.tech/dbname?sslmode=require

# Application settings
SECRET_KEY=your-secret-key-here
DEBUG=False
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # in seconds

# Ray cluster settings (optional)
RAY_CLUSTER_ADDRESS=ray://ray-head-svc:10001
```

### 2. Initialize the Vector Store
```bash
cd backend
python scripts/setup_qdrant.py
```

### 3. Run Database Migrations
```bash
cd backend
alembic upgrade head
```

## Running the System

### 1. Start Services with Docker Compose
```bash
docker-compose up -d
```

### 2. Or Run Services Separately

#### Start the Backend API:
```bash
cd backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Start the Frontend (Chainlit UI):
```bash
cd frontend
chainlit run src/pages/chat_ui.py -w
```

## Processing Books

### 1. Ingest a Book
```bash
cd backend
python scripts/ingest_content.py --file-path /path/to/book.pdf --book-title "Book Title" --book-author "Author Name"
```

### 2. Process Large Books with Ray (for books > 500 pages):
```bash
cd backend
python scripts/process_books.py --file-path /path/to/large_book.pdf --use-ray
```

## API Usage

### Chat Endpoint
Send a query to the chatbot:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main concept discussed in chapter 3?",
    "selected_text": "Optional selected text from the book viewer"
  }'
```

### Response Format
```json
{
  "answer": "The main concept discussed in chapter 3 is...",
  "citations": [
    {
      "chunk_id": "chunk-123",
      "page_number": 45,
      "section": "Chapter 3.2",
      "similarity_score": 0.87
    }
  ],
  "grounding_confidence": 0.92,
  "model_used": "gpt-4o"
}
```

## Frontend Integration

### 1. JavaScript Widget for Book Viewers
Add this script to your book viewer HTML:

```html
<script>
// Add event listener for text selection
document.addEventListener('mouseup', () => {
  let selected = window.getSelection().toString().trim();
  if (selected && selected.length > 10) { // Only send if meaningful text is selected
    fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: 'Explain this concept',
        selected_text: selected
      })
    })
    .then(response => response.json())
    .then(data => {
      // Display the response in your UI
      console.log('Chatbot response:', data.answer);
      // Example: Show in a popup or sidebar
      showChatbotResponse(data.answer, data.citations);
    })
    .catch(error => {
      console.error('Error getting chatbot response:', error);
    });
  }
});

function showChatbotResponse(answer, citations) {
  // Implement your UI to show the chatbot response
  // This could be a popup, sidebar, or inline element
}
</script>
```

### 2. Embedding the Chat Interface
To embed the full chat interface in your application:

```html
<iframe
  src="http://localhost:8000/chat-interface"
  width="400"
  height="600"
  frameborder="0">
</iframe>
```

## Testing

### 1. Run Unit Tests
```bash
cd backend
pytest tests/unit/ -v
```

### 2. Run Integration Tests
```bash
cd backend
pytest tests/integration/ -v
```

### 3. End-to-End Tests
```bash
cd backend
pytest tests/end_to_end.py -v
```

## Production Deployment

### 1. Kubernetes Deployment
Apply the Kubernetes manifests:

```bash
kubectl apply -f k8s/
kubectl apply -f k8s/dapr/
```

### 2. Dapr Sidecar Configuration
The system uses Dapr for actor state management in production. Make sure Dapr is installed in your Kubernetes cluster:

```bash
dapr init -k
```

### 3. Horizontal Pod Autoscaler
The system includes HPA configuration for automatic scaling:

```bash
kubectl apply -f .kubernetes/production/hpa.yaml
```

## Development

### 1. Adding New Agents
1. Create a new agent class in `backend/src/agents/`
2. Implement the required interface methods
3. Register the agent with the coordinator
4. Add unit tests in `tests/unit/test_agents/`

### 2. Extending Data Models
1. Update the data models in `backend/src/models/`
2. Create a new Alembic migration
3. Update the data-model.md documentation
4. Add or update validation rules

### 3. Frontend Development
The frontend includes:
- Chainlit UI for development and testing
- JavaScript widget for book viewer integration
- Embeddable components for different platforms

## Architecture Notes

- The system follows a multi-agent architecture with RetrievalAgent, GenerationAgent, and CoordinatorAgent
- All responses are grounded in book content to avoid hallucinations
- The system uses Qdrant for vector storage and Neon Postgres for metadata
- Rate limiting is implemented to prevent API abuse
- The system is designed for Kubernetes deployment with Dapr for actor state management
- Ray clusters are used for parallel processing of large books
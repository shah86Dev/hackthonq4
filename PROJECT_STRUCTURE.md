# Book-Embedded RAG Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) system designed for interactive textbook Q&A, featuring full-book and selected-text querying capabilities.

## 📚 Project Overview

This project implements a state-of-the-art RAG system that allows users to interact with book content through an AI-powered chatbot. The system supports both full-book queries and selected-text interactions, providing accurate answers with proper citations.

### Key Features
- **Full-Book RAG**: Query entire books for comprehensive answers
- **Selected-Text RAG**: Ask questions about specific highlighted text
- **Citation System**: Proper attribution to book sections/pages
- **Confidence Scoring**: Quality indicators for generated answers
- **Feedback Loop**: User rating system to improve responses
- **Multi-Modal**: Supports PDF and text content ingestion
- **Vector Storage**: Qdrant-powered semantic search
- **API-First**: RESTful API design for easy integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   Vector DB     │
│  (Docusaurus)   │───▶│    (FastAPI)     │───▶│   (Qdrant)      │
│                 │    │                  │    │                 │
│  - Chat UI      │    │  - API Gateway   │    │  - Semantic     │
│  - Book Viewer  │    │  - Query Logic   │    │    Search       │
│  - Citations    │    │  - Ingestion     │    │  - Embeddings   │
└─────────────────┘    │  - Authentication│    └─────────────────┘
                       │  - Logging       │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   Database       │
                       │                  │
                       │ - PostgreSQL     │
                       │ - Query Logs     │
                       │ - Answers        │
                       │ - Feedback       │
                       └──────────────────┘
```

## 📁 Project Structure

```
book-rag-chatbot/
├── backend/                 # FastAPI backend server
│   ├── src/
│   │   ├── api/            # API endpoints (ingest, query, health, chat)
│   │   ├── models/         # Database models (Book, Chunk, Answer, etc.)
│   │   ├── services/       # Business logic (Ingestion, Retrieval, Generation)
│   │   ├── config/         # Configuration settings
│   │   └── database.py     # Database connection
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/               # Docusaurus frontend
│   ├── src/
│   │   ├── components/     # React components (ChatBot, EmbeddedChatBot)
│   │   ├── pages/         # React pages
│   │   └── css/           # Styling
│   ├── docs/              # Book content and documentation
│   └── docusaurus.config.js # Frontend configuration
├── specs/                  # Project specifications
│   └── 003-book-rag-chatbot/
├── history/                # Project history and prompts
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-rag-chatbot
   ```

2. **Set up backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI and Qdrant keys
   ```

4. **Start backend server**
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Set up and start frontend**
   ```bash
   cd ../frontend
   npm install
   npm start
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔧 API Endpoints

### Ingestion
- `POST /api/v1/ingest/book` - Ingest a book into the RAG system

### Query
- `POST /api/v1/query` - Query book content (full-book or selected-text mode)

### Upload
- `POST /api/v1/upload/book` - Upload and process a book file

### Health Check
- `GET /api/v1/health` - Health check endpoint

## 💡 Usage Examples

### Query a Book
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "your-book-uuid",
    "question": "What is the main concept discussed in this book?",
    "mode": "full-book"
  }'
```

### Query with Selected Text
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "your-book-uuid",
    "question": "Explain this concept further",
    "mode": "selected-text",
    "selected_text": "The concept of embodied cognition..."
  }'
```

## 🤖 Technologies Used

- **Backend**: FastAPI, Python 3.11
- **Frontend**: Docusaurus, React, JavaScript
- **Database**: PostgreSQL (relational), Qdrant (vector)
- **AI/ML**: OpenAI API, Sentence Transformers
- **API Documentation**: Swagger/OpenAPI
- **Deployment**: Docker, Vercel

## 📊 Data Models

### QueryLog
- Tracks all user queries with metadata (book_id, mode, question, latency, tokens)

### Answer
- Stores generated answers with citations and confidence scores

### Feedback
- User ratings and corrections to improve response quality

## 🚀 Deployment

The application is designed for easy deployment to cloud platforms:

### Vercel (Frontend)
- Connect your GitHub repository to Vercel
- Set environment variables for backend URL

### Backend Deployment
- Deploy using Docker containers
- Configure with environment variables
- Set up SSL certificates for production

## 📈 Performance Metrics

- **Latency**: Query response times tracked in logs
- **Token Usage**: API cost optimization tracking
- **Confidence Scores**: Answer quality indicators
- **User Feedback**: Continuous improvement loop

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Project Goals

- Enable interactive learning through AI-powered textbook Q&A
- Provide accurate citations for all generated answers
- Support multiple query modes (full-book and selected-text)
- Create a feedback loop for continuous improvement
- Maintain high performance and reliability standards
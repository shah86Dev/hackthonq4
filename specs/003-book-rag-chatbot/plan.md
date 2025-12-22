# Implementation Plan: Book-Integrated RAG Chatbot

**Branch**: `003-book-rag-chatbot` | **Date**: 2025-12-20 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/003-book-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The Book-Integrated RAG Chatbot will implement a multi-agent architecture using OpenAI Agents SDK with RetrievalAgent, GenerationAgent, and CoordinatorAgent. The system will process book content through RecursiveCharacterTextSplitter → OpenAI embeddings → Qdrant vector store with Neon Postgres for metadata. The backend will use FastAPI with async endpoints, and the frontend will include a Chainlit UI and JavaScript snippet for book viewer integration. Deployment will leverage Docker Compose for local development and Kubernetes with Dapr for production.

## Technical Context

**Language/Version**: Python 3.11 (for backend services and data processing), JavaScript (for frontend integration)
**Primary Dependencies**: FastAPI, OpenAI SDK, Qdrant Client, Neon Postgres (psycopg2), LangChain, Chainlit, OpenAI Agents SDK
**Storage**: Qdrant vector database for embeddings, Neon Postgres for metadata and session data
**Testing**: pytest for backend services, integration tests for agent workflows, unit tests for core functions
**Target Platform**: Linux server environment (cloud deployment), cross-browser compatible for frontend integration
**Project Type**: Web-based service with embedded frontend component
**Performance Goals**: <2s response time for 95% of queries (per constitutional requirement), 90% accuracy in content retrieval, handle books up to 1000 pages efficiently
**Constraints**: Must avoid hallucinations (per constitutional requirement), implement rate limiting, ensure token-efficient processing (per constitutional requirement), secure API key management
**Scale/Scope**: Support concurrent user sessions, handle large books (>500 pages) with Ray parallel processing, 99.5% uptime requirement

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

1. **Accuracy-First Design**: ✅
   - Responses will be restricted to book content or user-selected text only
   - All responses will be grounded in provided source material with proper citation
   - Hallucinations will be avoided through strict context limitations

2. **Scalability Architecture**: ✅
   - Using Qdrant Cloud Free Tier for vector storage
   - Ray for distributed embedding processing
   - Designed to handle growing content and user load

3. **Security-First Integration**: ✅
   - API keys for OpenAI, Neon Postgres, and Qdrant
   - FastAPI with rate limiting implementation
   - Proper authentication and authorization measures

4. **User Experience Focus**: ✅
   - JavaScript text selection support in book viewers
   - Fallback to full-book retrieval when needed
   - Intuitive access to both selected-text and full-book query modes

5. **Testing Excellence**: ✅
   - Unit tests for chunking, embedding, retrieval, and generation
   - Automated evaluations for response relevance and accuracy
   - Comprehensive test coverage planned

6. **Performance Optimization**: ✅
   - <2s response time target (constitutional requirement)
   - Efficient handling of large books (>500 pages)
   - Optimized for low-latency queries

7. **Kubernetes-Ready Deployment**: ✅
   - Prepared for Kubernetes deployment with Dapr sidecars
   - Production-ready scalability and reliability features
   - Agent state management with Dapr

8. **Artifact Traceability**: ✅
   - All specs, prompt history, and architecture decisions in Git
   - Versioned artifacts for complete traceability
   - Proper documentation and audit trails

9. **Token-Efficient Architecture**: ✅
   - Following constitutional requirement for token-efficient processing
   - Heavy logic will execute in scripts, not in LLM context
   - Only skill instructions and final outputs will enter context

### Potential Violations and Justifications

- No violations identified. All architectural decisions align with constitutional principles.

### Post-Design Constitution Re-check

After implementing the detailed design:

1. **Accuracy-First Design**: ✅ Confirmed
   - API enforces responses are grounded in book content only
   - Response model includes grounding_confidence metric
   - Citations are provided for all referenced content

2. **Scalability Architecture**: ✅ Confirmed
   - Qdrant vector store supports cloud deployment
   - API supports batch processing for large books
   - Rate limiting prevents resource exhaustion

3. **Security-First Integration**: ✅ Confirmed
   - API endpoints follow security best practices
   - Rate limiting implemented to prevent abuse
   - Sensitive data properly handled in models

4. **User Experience Focus**: ✅ Confirmed
   - API supports both full-book and selected-text queries
   - Rich response model with citations and context
   - Session management for conversation continuity

5. **Testing Excellence**: ✅ Confirmed
   - API contract defines clear interfaces for testing
   - Response models include testable metrics
   - Error handling patterns are standardized

6. **Performance Optimization**: ✅ Confirmed
   - Response times measured and reported in API
   - Chunked content enables efficient retrieval
   - Health check endpoint monitors service performance

7. **Kubernetes-Ready Deployment**: ✅ Confirmed
   - API designed for containerized deployment
   - External service dependencies (Qdrant, Postgres) properly abstracted
   - Configuration through environment variables

8. **Artifact Traceability**: ✅ Confirmed
   - API contract documented in OpenAPI format
   - Data models fully specified with validation rules
   - Implementation follows constitutional requirements

9. **Token-Efficient Architecture**: ✅ Confirmed
   - API minimizes response sizes with efficient data models
   - Only necessary context is passed between components
   - Heavy processing happens in dedicated services, not in API layer

## Project Structure

### Documentation (this feature)

```text
specs/003-book-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── agents/              # Agent implementations (RetrievalAgent, GenerationAgent, CoordinatorAgent)
│   │   ├── __init__.py
│   │   ├── retrieval_agent.py
│   │   ├── generation_agent.py
│   │   └── coordinator_agent.py
│   ├── api/                 # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── chat_routes.py
│   ├── core/                # Core logic and utilities
│   │   ├── __init__.py
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter implementation
│   │   ├── embedder.py      # OpenAI embedding functions
│   │   ├── vector_store.py  # Qdrant integration
│   │   └── database.py      # Neon Postgres integration
│   ├── models/              # Pydantic models and data structures
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   └── response_models.py
│   └── config/              # Configuration settings
│       ├── __init__.py
│       └── settings.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_agents/
│   │   ├── test_core/
│   │   └── test_api/
│   ├── integration/
│   │   └── test_end_to_end.py
│   └── conftest.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── alembic/
    ├── alembic.ini
    └── versions/
        └── 001_initial_schema.py

frontend/
├── src/
│   ├── components/
│   │   └── rag_chat_widget.js  # JS snippet for book viewer integration
│   └── pages/
│       └── chat_ui.py          # Chainlit chat interface
├── public/
│   └── embed-script.js         # Standalone embedding script
└── index.html
└── package.json

scripts/
├── ingest_content.py           # Content ingestion script
├── process_books.py            # Batch processing with Ray
└── setup_qdrant.py             # Vector store initialization

k8s/
├── deployment.yaml
├── service.yaml
├── ingress.yaml
└── dapr/
    ├── components/
    │   ├── statestore.yaml
    │   └── pubsub.yaml
    └── configurations/
        └── appconfig.yaml

ray/
├── cluster-config.yaml
└── processing_jobs.py

.devcontainer/
└── devcontainer.json

.kubernetes/
└── production/
    ├── namespace.yaml
    ├── secrets.yaml
    ├── configmaps.yaml
    ├── deployment.yaml
    ├── service.yaml
    └── hpa.yaml
```

**Structure Decision**: Multi-service architecture with separate backend for API/agents and frontend for UI components. The backend implements the multi-agent RAG system with proper separation of concerns. The frontend provides both a testing UI (Chainlit) and embeddable components (JavaScript widget) for book viewer integration. This structure supports the constitutional requirements for scalability, security, and performance while maintaining clean separation between components.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [No violations] | [All architectural decisions comply with constitution] | [N/A] |

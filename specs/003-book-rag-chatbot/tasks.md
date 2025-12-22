# Implementation Tasks: Book-Integrated RAG Chatbot

**Feature**: Book-Integrated RAG Chatbot | **Date**: 2025-12-20 | **Branch**: 003-book-rag-chatbot

## Overview

This document outlines the granular tasks for implementing the Book-Integrated RAG Chatbot. The implementation follows a multi-agent architecture with RetrievalAgent, GenerationAgent, and CoordinatorAgent, using FastAPI backend, Qdrant vector store, and Neon Postgres for metadata.

## Dependencies

- Python 3.11+ with pip
- Docker and Docker Compose
- Access to OpenAI API (with text-embedding-ada-002 and GPT-4o access)
- Qdrant Cloud Free Tier account
- Neon Postgres account
- Node.js 18+ for frontend components

## Parallel Execution Examples

- T001-T003 (Environment setup) can run in parallel with T004-T006 (Project structure)
- US1 tasks (Book Content Query) can run in parallel with US2 tasks (Selected Text Interaction) after foundational components are built
- Backend development (T010+) can run in parallel with frontend development (T070+)

## Implementation Strategy

- MVP First: Complete User Story 1 (Book Content Query) with minimal viable functionality
- Incremental Delivery: Add User Stories 2 and 3 in sequence
- Test-Driven Development: Write unit tests alongside implementation
- Continuous Integration: Each completed user story should be independently testable

---

## Phase 1: Setup Tasks

**Goal**: Establish development environment and project structure

- [ ] T001 Set up Python virtual environment in backend/ with Python 3.11
- [ ] T002 Install dependencies: fastapi, uvicorn, openai, qdrant-client, psycopg2-binary, langchain, chainlit
- [ ] T003 Install additional dependencies: python-multipart, pydantic, python-dotenv, asyncpg
- [ ] T004 Create project structure per implementation plan in backend/
- [ ] T005 Create project structure per implementation plan in frontend/
- [ ] T006 Create scripts/ directory with ingestion script template
- [ ] T007 Initialize git repository with proper .gitignore for Python/JavaScript projects
- [ ] T008 Set up requirements.txt with pinned versions of all dependencies
- [ ] T009 Create .env.example with all required environment variables

---

## Phase 2: Foundational Tasks

**Goal**: Build core components that all user stories depend on

- [ ] T010 [P] Create configuration module in backend/src/config/settings.py with Pydantic BaseSettings
- [ ] T011 [P] Implement Qdrant vector store integration in backend/src/core/vector_store.py
- [ ] T012 [P] Implement Neon Postgres database integration in backend/src/core/database.py
- [ ] T013 [P] Create chunking utility using RecursiveCharacterTextSplitter in backend/src/core/chunker.py
- [ ] T014 [P] Create embedding utility using OpenAI API in backend/src/core/embedder.py
- [ ] T015 [P] Define request/response models in backend/src/models/request_models.py
- [ ] T016 [P] Define response models in backend/src/models/response_models.py
- [ ] T017 [P] Set up basic FastAPI application structure in backend/src/api/main.py
- [ ] T018 [P] Implement rate limiting middleware for API protection
- [ ] T019 Create ingestion script structure in scripts/ingest_content.py
- [ ] T020 Set up Alembic for database migrations in backend/alembic/

---

## Phase 3: User Story 1 - Book Content Query (Priority: P1)

**Goal**: Enable readers to ask questions about book content and receive accurate responses based on the book's information

**Independent Test**: Can be fully tested by uploading sample book content, asking questions, and verifying that responses are grounded in the book content and accurate.

**Acceptance Scenarios**:
1. Given a digital book with processed content in the system, When a user submits a question about the book, Then the system retrieves relevant content and generates an accurate answer based only on the book material
2. Given a user query that matches specific book content, When the query is processed through the RAG pipeline, Then the system returns a response that cites the relevant book sections

### Tests (if requested)
- [ ] T021 [US1] Write unit tests for chunking functionality in backend/tests/unit/test_core/test_chunker.py
- [ ] T022 [US1] Write unit tests for embedding functionality in backend/tests/unit/test_core/test_embedder.py
- [ ] T023 [US1] Write unit tests for vector store operations in backend/tests/unit/test_core/test_vector_store.py

### Models
- [ ] T024 [US1] Extend BookContentChunk model with proper validation in backend/src/models/chunk_models.py
- [ ] T025 [US1] Create BookMetadata model in backend/src/models/book_models.py

### Services
- [ ] T026 [US1] Implement RetrievalAgent class in backend/src/agents/retrieval_agent.py
- [ ] T027 [US1] Implement embedding query and Qdrant search in RetrievalAgent
- [ ] T028 [US1] Implement fetching top-5 chunks with metadata from Postgres in RetrievalAgent
- [ ] T029 [US1] Implement GenerationAgent class in backend/src/agents/generation_agent.py
- [ ] T030 [US1] Integrate OpenAI Assistant with RAG instructions in GenerationAgent
- [ ] T031 [US1] Implement context threading with retrieved content in GenerationAgent
- [ ] T032 [US1] Implement CoordinatorAgent class in backend/src/agents/coordinator_agent.py
- [ ] T033 [US1] Implement API request handling in CoordinatorAgent
- [ ] T034 [US1] Implement fallback to full-book retrieval when selected_text is empty

### Endpoints
- [ ] T035 [US1] Create /chat endpoint in backend/src/api/chat_routes.py
- [ ] T036 [US1] Integrate agents with /chat endpoint
- [ ] T037 [US1] Implement query validation and error handling in /chat endpoint
- [ ] T038 [US1] Add grounding confidence calculation to response
- [ ] T039 [US1] Add citation information to response

### Integration
- [ ] T040 [US1] Write integration test for full query flow in backend/tests/integration/test_book_query.py
- [ ] T041 [US1] Test end-to-end functionality with sample book content

---

## Phase 4: User Story 2 - Selected Text Interaction (Priority: P2)

**Goal**: Allow readers to select specific text in the digital book viewer and ask questions about that particular selection

**Independent Test**: Can be tested by selecting text in a book viewer and verifying that the chatbot uses that specific text as context for responses.

**Acceptance Scenarios**:
1. Given a user has selected text in the book viewer, When they initiate a chat with that selection, Then the system uses the selected text as primary context for the response
2. Given a user query about selected text, When the query is processed, Then the response is specifically tailored to the selected content

### Services
- [ ] T042 [US2] Enhance CoordinatorAgent to handle selected_text parameter
- [ ] T043 [US2] Modify RetrievalAgent to prioritize selected_text when provided
- [ ] T044 [US2] Update GenerationAgent to use selected_text as primary context
- [ ] T045 [US2] Implement fallback mechanism when selected_text is empty

### Endpoints
- [ ] T046 [US2] Update /chat endpoint to properly handle selected_text parameter
- [ ] T047 [US2] Add validation for selected_text parameter
- [ ] T048 [US2] Implement response differentiation when selected_text is used

### Frontend
- [ ] T049 [US2] Create Chainlit chat UI in frontend/src/pages/chat_ui.py
- [ ] T050 [US2] Add support for selected_text metadata in Chainlit interface

### Integration
- [ ] T051 [US2] Write integration test for selected text functionality in backend/tests/integration/test_selected_text.py

---

## Phase 5: User Story 3 - Embedded Widget Integration (Priority: P3)

**Goal**: Enable publishers to embed the RAG chatbot into their web-based book platform seamlessly

**Independent Test**: Can be tested by embedding the chatbot widget in a sample web page and verifying functionality.

**Acceptance Scenarios**:
1. Given a web-based book platform, When the RAG chatbot widget is embedded, Then the widget functions properly and integrates with the book content
2. Given an embedded chatbot widget, When a user interacts with it, Then the widget communicates properly with the backend API

### Frontend
- [ ] T052 [US3] Create JavaScript widget for book viewer integration in frontend/src/components/rag_chat_widget.js
- [ ] T053 [US3] Implement text selection detection using window.getSelection() in rag_chat_widget.js
- [ ] T054 [US3] Add event listener for mouseup to detect text selection in rag_chat_widget.js
- [ ] T055 [US3] Implement API communication for selected text in rag_chat_widget.js
- [ ] T056 [US3] Create standalone embedding script in frontend/public/embed-script.js
- [ ] T057 [US3] Add iframe support for chat interface in frontend/index.html

### Backend
- [ ] T058 [US3] Add CORS support for cross-origin embedding in FastAPI
- [ ] T059 [US3] Implement health check endpoint in backend/src/api/main.py
- [ ] T060 [US3] Add API response formatting suitable for frontend widgets

### Integration
- [ ] T061 [US3] Write integration test for embedded widget functionality in backend/tests/integration/test_widget_integration.py
- [ ] T062 [US3] Test widget functionality across different browsers

---

## Phase 6: Testing & Evaluation

**Goal**: Implement comprehensive tests and evaluation mechanisms

- [ ] T063 Write unit tests for ingestion script functionality in backend/tests/unit/test_ingestion.py
- [ ] T064 Write unit tests for all agent functionalities in backend/tests/unit/test_agents/
- [ ] T065 Implement automated evals comparing responses against ground-truth Q&A in backend/tests/evaluations/
- [ ] T066 Set up pytest configuration in backend/pytest.ini
- [ ] T067 Add test coverage reporting in backend/.coveragerc

---

## Phase 7: Deployment & Infrastructure

**Goal**: Prepare the application for deployment with Docker, Kubernetes, and Ray

- [ ] T068 Create Dockerfile for FastAPI backend in backend/Dockerfile
- [ ] T069 Create docker-compose.yml for local development in backend/docker-compose.yml
- [ ] T070 Create Kubernetes deployment manifests in k8s/
- [ ] T071 Add Dapr configuration for production in k8s/dapr/
- [ ] T072 Create Ray job configuration for ingestion in ray/
- [ ] T073 Update ingestion script to support Ray parallel processing in scripts/ingest_content.py
- [ ] T074 Create production Kubernetes manifests in .kubernetes/production/

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Final touches, documentation, and optimization

- [ ] T075 Add logging throughout the application in backend/src/core/logging.py
- [ ] T076 Implement error handling and user-friendly messages
- [ ] T077 Add monitoring and metrics endpoints
- [ ] T078 Create comprehensive API documentation using FastAPI's automatic documentation
- [ ] T079 Write user documentation for embedding the widget
- [ ] T080 Perform performance optimization and load testing
- [ ] T081 Add security headers and implement security best practices
- [ ] T082 Create deployment scripts and CI/CD pipeline configuration
- [ ] T083 Final testing and bug fixes
- [ ] T084 Prepare release notes and documentation
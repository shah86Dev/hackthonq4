# Tasks: Book-Embedded RAG Chatbot

**Feature**: Book-Embedded RAG Chatbot
**Branch**: 004-book-rag-chatbot
**Created**: 2025-12-25
**Input**: Feature specification and architecture plan from `/specs/004-book-rag-chatbot/`

## Implementation Strategy

This task breakdown implements a multi-agent RAG chatbot system with FastAPI backend, Qdrant vector store, and Neon Postgres metadata storage. The approach follows the user story priorities (P1, P2, P3) to enable independent implementation and testing of each feature. Tasks are organized to create an MVP with User Story 1 first, then incrementally add functionality for other stories.

## Dependencies

User stories have the following dependency relationships:
- User Story 1 (P1) - Core functionality, no dependencies
- User Story 2 (P2) - Depends on User Story 1 for basic chat functionality
- User Story 3 (P3) - Depends on User Story 1 for ingestion pipeline
- User Story 4 (P3) - Depends on User Story 3 for basic processing, and User Story 1 for core functionality

## Parallel Execution Examples

Each user story includes parallelizable tasks (marked with [P]) that can be executed simultaneously by different agents:
- US1: Model creation, service implementation, and endpoint development can proceed in parallel
- US2: Frontend components can be developed in parallel with backend enhancements
- US3: Format-specific processors can be developed in parallel

## Phase 1: Setup

Goal: Establish project infrastructure and dependencies

- [ ] T001 Create project structure with backend, frontend, and shared directories per implementation plan
- [ ] T002 [P] Set up environment with dependencies: fastapi, uvicorn, openai, qdrant-client, psycopg2, langchain, chainlit
- [ ] T003 [P] Initialize backend requirements.txt with all required dependencies
- [ ] T004 [P] Initialize frontend package.json with necessary dependencies
- [ ] T005 Create shared docker-compose.yml with all services
- [ ] T006 Set up configuration files for API keys and service connections
- [ ] T007 Create basic testing structure with pytest configuration

## Phase 2: Foundational

Goal: Core infrastructure and data models needed for all user stories

- [ ] T008 [P] Create Book Chunk model in backend/src/models/chunk.py with validation rules
- [ ] T009 [P] Create Query Log model in backend/src/models/query_log.py with validation rules
- [ ] T010 [P] Create Book Metadata model in backend/src/models/book_metadata.py with validation rules
- [ ] T011 [P] Implement Qdrant client connection in backend/src/services/qdrant_client.py
- [ ] T012 [P] Implement Neon Postgres connection in backend/src/database.py
- [ ] T013 [P] Create configuration settings in backend/src/config/settings.py
- [ ] T014 [P] Implement basic middleware for rate limiting in backend/src/api/middleware.py
- [ ] T015 [P] Create API response models in backend/src/schemas.py

## Phase 3: User Story 1 - Query Book Content via Chatbot (Priority: P1)

Goal: Enable users to ask questions about book content and receive accurate answers based on book content

**Independent Test**: Can be fully tested by providing a book, asking questions about its content, and verifying that responses are accurate and based on the book's text. This delivers the core value of enhanced book interaction.

- [ ] T016 [P] [US1] Implement text extraction from PDF in backend/src/services/ingestion_service.py
- [ ] T017 [P] [US1] Implement text extraction from Markdown in backend/src/services/ingestion_service.py
- [ ] T018 [P] [US1] Implement chunking service using RecursiveCharacterTextSplitter in backend/src/services/ingestion_service.py
- [ ] T019 [P] [US1] Implement embedding generation using OpenAI API in backend/src/services/ingestion_service.py
- [ ] T020 [P] [US1] Implement Qdrant vector storage for chunks in backend/src/services/ingestion_service.py
- [ ] T021 [P] [US1] Implement Neon Postgres metadata storage in backend/src/services/ingestion_service.py
- [ ] T022 [P] [US1] Implement RetrievalAgent to search Qdrant for top-5 chunks in backend/src/agents/retrieval_agent.py
- [ ] T023 [P] [US1] Implement GenerationAgent with OpenAI Assistant (GPT-4o) in backend/src/agents/generation_agent.py
- [ ] T024 [P] [US1] Implement CoordinatorAgent to handle API logic in backend/src/agents/coordinator_agent.py
- [ ] T025 [US1] Create POST /chat endpoint in backend/src/api/endpoints/chat.py
- [ ] T026 [US1] Implement basic Chainlit chat UI in frontend/src/chat_ui.py
- [ ] T027 [US1] Write unit tests for chunking service in backend/tests/unit/test_chunking.py
- [ ] T028 [US1] Write unit tests for embedding service in backend/tests/unit/test_embedding.py
- [ ] T029 [US1] Write integration tests for full query flow in backend/tests/integration/test_query_flow.py
- [ ] T030 [US1] Test acceptance scenario: Given a book has been processed and embedded in the system, When a user submits a question via the chat interface, Then the system returns an answer based on the book content with proper context citations
- [ ] T031 [US1] Test acceptance scenario: Given a user has asked a question about the book content, When the system processes the query, Then the response is grounded in the book's content without hallucinations

## Phase 4: User Story 2 - Query Selected Text Context (Priority: P2)

Goal: Allow users to select specific text in the book viewer and ask questions about that specific passage

**Independent Test**: Can be tested by selecting text in the book viewer, asking a question, and verifying that the system prioritizes the selected text as context for the answer. This delivers value by allowing focused questioning of specific content.

**Dependencies**: Requires User Story 1 for basic chat functionality

- [ ] T032 [P] [US2] Enhance CoordinatorAgent to handle selected_text parameter in backend/src/agents/coordinator_agent.py
- [ ] T033 [P] [US2] Modify RetrievalAgent to prioritize selected text context in backend/src/agents/retrieval_agent.py
- [ ] T034 [P] [US2] Update GenerationAgent to use selected text as primary context in backend/src/agents/generation_agent.py
- [ ] T035 [US2] Update POST /chat endpoint to accept selected_text parameter in backend/src/api/endpoints/chat.py
- [ ] T036 [US2] Implement JavaScript text selection functionality in frontend/src/components/FloatingChatButton/text_selection.js
- [ ] T037 [US2] Update Chainlit chat UI to support selected text context in frontend/src/chat_ui.py
- [ ] T038 [US2] Test acceptance scenario: Given a reader has selected text in the book viewer, When they ask a question, Then the system uses the selected text as primary context for the answer
- [ ] T039 [US2] Test acceptance scenario: Given a reader has selected text and asked a related question, When the system generates the response, Then the answer is more focused on the selected content than general book knowledge

## Phase 5: User Story 3 - Process Different Book Formats (Priority: P3)

Goal: Enable processing of different book formats (PDF, Markdown) for RAG functionality

**Independent Test**: Can be tested by uploading different book formats (PDF, Markdown), verifying they're properly processed and chunked, and confirming that queries work on the processed content. This delivers value by supporting multiple content types.

**Dependencies**: Requires User Story 1 for ingestion pipeline

- [ ] T040 [P] [US3] Implement comprehensive PDF text extraction in backend/src/services/ingestion_service.py
- [ ] T041 [P] [US3] Implement comprehensive Markdown text extraction in backend/src/services/ingestion_service.py
- [ ] T042 [P] [US3] Add TXT format support in backend/src/services/ingestion_service.py
- [ ] T043 [P] [US3] Create format detection utility in backend/src/services/ingestion_service.py
- [ ] T044 [P] [US3] Implement POST /ingest endpoint in backend/src/api/endpoints/ingest.py
- [ ] T045 [US3] Write unit tests for PDF extraction in backend/tests/unit/test_pdf_extraction.py
- [ ] T046 [US3] Write unit tests for Markdown extraction in backend/tests/unit/test_markdown_extraction.py
- [ ] T047 [US3] Test acceptance scenario: Given a PDF book file, When the system processes it, Then the text is extracted and prepared for RAG functionality
- [ ] T048 [US3] Test acceptance scenario: Given a Markdown book file, When the system processes it, Then the content is properly structured for embedding and retrieval

## Phase 6: User Story 4 - Scalable Book Processing (Priority: P3)

Goal: Process large books (>500 pages) efficiently using distributed methods without performance degradation

**Independent Test**: Can be tested by processing a large book (>500 pages), verifying it completes without errors, and confirming query performance remains acceptable. This delivers value by supporting comprehensive content.

**Dependencies**: Requires User Story 3 for basic processing and User Story 1 for core functionality

- [ ] T049 [P] [US4] Implement Ray cluster integration for distributed embedding in backend/src/services/ingestion_service.py
- [ ] T050 [P] [US4] Create distributed chunking function using Ray in backend/src/services/ingestion_service.py
- [ ] T051 [P] [US4] Implement batch processing for large books in backend/src/services/ingestion_service.py
- [ ] T052 [P] [US4] Add performance monitoring for large book processing in backend/src/services/ingestion_service.py
- [ ] T053 [P] [US4] Implement rate limiting for ingestion API in backend/src/api/endpoints/ingest.py
- [ ] T054 [US4] Test acceptance scenario: Given a large book (>500 pages), When the system processes it, Then the processing completes efficiently using distributed methods
- [ ] T055 [US4] Test acceptance scenario: Given a large processed book, When users query it, Then response times remain under 2 seconds

## Phase 7: User Story 5 - Frontend Integration and Deployment

Goal: Enable seamless integration of the chatbot into web-based book platforms

**Dependencies**: Requires User Story 1 for core functionality

- [ ] T056 [P] [US5] Create embeddable chat widget component in frontend/src/components/EmbeddedChatBot/index.js
- [ ] T057 [P] [US5] Implement JavaScript snippet for book embedding in frontend/src/utils/embedding.js
- [ ] T058 [P] [US5] Create iframe wrapper for chat widget in frontend/src/components/ChatBot/index.js
- [ ] T059 [P] [US5] Implement window.getSelection() integration in frontend/src/utils/text_selection.js
- [ ] T060 [P] [US5] Create Dockerfile for FastAPI backend
- [ ] T061 [P] [US5] Create Dockerfile for frontend
- [ ] T062 [P] [US5] Create Kubernetes deployment manifests in shared/k8s/deployment.yaml
- [ ] T063 [P] [US5] Create Kubernetes service manifests in shared/k8s/service.yaml
- [ ] T064 [P] [US5] Create Dapr configuration in shared/dapr/config.yaml
- [ ] T065 [P] [US5] Create Ray job configuration for ingestion in shared/ray/ingestion_job.yaml
- [ ] T066 [US5] Test frontend integration with book viewer

## Phase 8: User Story 6 - Testing and Evaluation

Goal: Implement comprehensive testing and automated evaluation framework

**Dependencies**: All previous user stories

- [ ] T067 [P] [US6] Write contract tests for API endpoints in backend/tests/contract/
- [ ] T068 [P] [US6] Create automated evaluation framework in backend/src/evaluation/evaluator.py
- [ ] T069 [P] [US6] Implement ground-truth Q&A comparison in backend/src/evaluation/evaluator.py
- [ ] T070 [P] [US6] Create test datasets from book samples in backend/tests/data/
- [ ] T071 [P] [US6] Implement response quality metrics in backend/src/evaluation/metrics.py
- [ ] T072 [P] [US6] Add comprehensive logging for evaluation in backend/src/services/logging.py
- [ ] T073 [US6] Run automated evaluations comparing responses against ground-truth Q&A
- [ ] T074 [US6] Implement performance benchmarks for response times
- [ ] T075 [US6] Create evaluation reports and dashboards

## Phase 9: Polish & Cross-Cutting Concerns

Goal: Final touches and cross-cutting concerns for production readiness

- [ ] T076 [P] Add comprehensive error handling throughout the application
- [ ] T077 [P] Implement proper logging and monitoring
- [ ] T078 [P] Add API documentation with OpenAPI/Swagger
- [ ] T079 [P] Implement health check endpoint in backend/src/api/endpoints/health.py
- [ ] T080 [P] Add comprehensive input validation
- [ ] T081 [P] Implement proper authentication and authorization
- [ ] T082 [P] Add security headers and protections
- [ ] T083 [P] Create comprehensive README and documentation
- [ ] T084 [P] Add CI/CD pipeline configuration
- [ ] T085 Conduct final integration testing across all components
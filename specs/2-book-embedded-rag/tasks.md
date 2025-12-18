---
description: "Task list for Book-Embedded RAG Chatbot implementation"
---

# Tasks: Book-Embedded RAG Chatbot

**Input**: Design documents from `/specs/2-book-embedded-rag/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `chatbot/src/`
- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `chatbot/src/`, `chatbot/dist/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan in backend/ and chatbot/
- [ ] T002 Initialize Python project with FastAPI, OpenAI, Qdrant, and Neon dependencies in backend/
- [ ] T003 [P] Initialize JavaScript project with build tools in chatbot/
- [ ] T004 [P] Configure linting and formatting tools for Python and JavaScript

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Setup database schema and migrations framework for Neon Postgres
- [ ] T006 [P] Configure Qdrant client and collection setup
- [ ] T007 [P] Setup API routing and middleware structure in backend/src/api/
- [ ] T008 Create base models/entities that all stories depend on in backend/src/models/
- [ ] T009 Configure error handling and logging infrastructure
- [ ] T010 Setup environment configuration management in backend/src/config/
- [ ] T011 [P] Setup OpenAI client and embedding service in backend/src/services/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Q&A from Book Content (Priority: P1) 🎯 MVP

**Goal**: Enable readers to ask questions about book content and get answers based solely on the book's text with citations

**Independent Test**: Ask questions about book content and verify answers are grounded in the book's text with proper citations

### Tests for User Story 1 (REQUIRED) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T012 [P] [US1] Contract test for ingestion API in backend/tests/contract/test_ingestion.py
- [ ] T013 [P] [US1] Contract test for full-book query API in backend/tests/contract/test_query.py
- [ ] T014 [P] [US1] Integration test for complete user journey in backend/tests/integration/test_basic_qa.py

### Implementation for User Story 1

- [ ] T015 [P] [US1] Create Book model in backend/src/models/book.py
- [ ] T016 [P] [US1] Create Chunk model in backend/src/models/chunk.py
- [ ] T017 [P] [US1] Create Query model in backend/src/models/query_log.py
- [ ] T018 [P] [US1] Create Answer model in backend/src/models/answer.py
- [ ] T019 [US1] Implement ingestion service in backend/src/services/ingestion_service.py
- [ ] T020 [US1] Implement retrieval service in backend/src/services/retrieval_service.py
- [ ] T021 [US1] Implement generation service in backend/src/services/generation_service.py
- [ ] T022 [US1] Implement citation service in backend/src/services/citation_service.py
- [ ] T023 [US1] Implement ingestion endpoint in backend/src/api/endpoints/ingest.py
- [ ] T024 [US1] Implement full-book query endpoint in backend/src/api/endpoints/query.py
- [ ] T025 [US1] Add validation and error handling for grounded responses
- [ ] T026 [US1] Add logging for query operations and observability
- [ ] T027 [US1] Implement confidence scoring mechanism
- [ ] T028 [US1] Add book isolation to prevent cross-book leakage

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Selected Text-Only Mode (Priority: P2)

**Goal**: Enable readers to highlight specific text and get answers based only on that highlighted content

**Independent Test**: Highlight text, ask questions, and verify answers are derived exclusively from the highlighted text without accessing broader book content

### Tests for User Story 2 (REQUIRED) ⚠️

- [ ] T029 [P] [US2] Contract test for selected-text query API in backend/tests/contract/test_selected_text.py
- [ ] T030 [P] [US2] Integration test for selected-text user journey in backend/tests/integration/test_selected_text.py

### Implementation for User Story 2

- [ ] T031 [P] [US2] Extend query endpoint to support selected-text mode in backend/src/api/endpoints/query.py
- [ ] T032 [US2] Implement selected-text retrieval logic in backend/src/services/retrieval_service.py
- [ ] T033 [US2] Add validation to ensure selected-text mode bypasses vector DB
- [ ] T034 [US2] Update citation service to handle selected-text citations in backend/src/services/citation_service.py
- [ ] T035 [US2] Add selected-text specific logging and observability

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Question Context and Citations (Priority: P3)

**Goal**: Provide clear citations for every answer so users can verify the source of information

**Independent Test**: Ask questions and verify every answer includes specific citations to book sections/chapters/pages

### Tests for User Story 3 (REQUIRED) ⚠️

- [ ] T036 [P] [US3] Contract test for citation format in backend/tests/contract/test_citations.py
- [ ] T037 [P] [US3] Integration test for citation accuracy in backend/tests/integration/test_citations.py

### Implementation for User Story 3

- [ ] T038 [P] [US3] Enhance citation service with detailed metadata extraction in backend/src/services/citation_service.py
- [ ] T039 [US3] Update answer formatting to include proper citation structure
- [ ] T040 [US3] Implement citation validation to ensure accuracy
- [ ] T041 [US3] Add citation tracking to query logging

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Frontend Embed Implementation

**Goal**: Create lightweight JavaScript widget for embedding in digital books

- [ ] T042 [P] Create core RAG engine in chatbot/src/core/rag_engine.js
- [ ] T043 [P] Implement mode resolver in chatbot/src/core/mode_resolver.js
- [ ] T044 [P] Create citation formatter in chatbot/src/core/citation_formatter.js
- [ ] T045 Create embed widget in chatbot/src/embed/widget.js
- [ ] T046 Implement highlight capture functionality in chatbot/src/embed/highlight_capture.js
- [ ] T047 Create chat UI component in chatbot/src/embed/chat_ui.js
- [ ] T048 Implement API client in chatbot/src/api/client.js
- [ ] T049 Build distributable widget in chatbot/dist/book-embedded-rag.js
- [ ] T050 Add frontend tests for widget functionality in chatbot/tests/

---

## Phase 7: Security and Observability

**Goal**: Implement security measures and comprehensive observability

- [ ] T051 [P] Implement rate limiting per session
- [ ] T052 [P] Add prompt injection detection and sanitization
- [ ] T053 Add comprehensive query logging to Postgres
- [ ] T054 Implement health check endpoint in backend/src/api/endpoints/health.py
- [ ] T055 Add performance monitoring and metrics

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T056 [P] Documentation updates in docs/
- [ ] T057 Code cleanup and refactoring
- [ ] T058 Performance optimization across all stories
- [ ] T059 [P] Additional unit tests in backend/tests/unit/ and chatbot/tests/
- [ ] T060 Security hardening
- [ ] T061 Run quickstart.md validation
- [ ] T062 Final integration testing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Frontend Implementation (Phase 6)**: Depends on API endpoints being available
- **Security/Observability (Phase 7)**: Can run in parallel with user stories
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds upon US1 infrastructure
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Enhances US1/US2 with citations

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories
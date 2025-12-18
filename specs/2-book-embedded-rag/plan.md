# Implementation Plan: Book-Embedded RAG Chatbot

**Branch**: `2-book-embedded-rag` | **Date**: 2025-12-17 | **Spec**: [specs/2-book-embedded-rag/spec.md](specs/2-book-embedded-rag/spec.md)
**Input**: Feature specification from `/specs/2-book-embedded-rag/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot that is embedded directly into digital books. The system will answer user questions strictly from the book's content, with support for both full-book RAG and selected-text-only RAG modes. The solution includes ingestion, retrieval, generation, and frontend embedding components with observability and security measures.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend embed
**Primary Dependencies**: FastAPI, OpenAI Agents SDK/ChatKit, Qdrant, Neon Postgres, OpenAI text-embedding models
**Storage**: Qdrant Cloud (vector database), Neon Serverless Postgres (relational data)
**Testing**: pytest for backend, Jest for frontend components
**Target Platform**: Web-based digital books (Docusaurus, HTML, EPUB WebView)
**Project Type**: Web application with embedded frontend component
**Performance Goals**: <2s P95 latency for common queries, support 10k+ pages and concurrent users
**Constraints**: Zero hallucinations, citation mandatory, book-level isolation, swappable components
**Scale/Scope**: Support 10,000+ book pages, concurrent users, multiple books simultaneously

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Spec-first development: All code derived from approved specifications
- Grounded responses: System answers strictly from retrieved textbook content
- Citation mandatory: Every answer includes page/section references
- Determinism: Same input + same corpus = same output
- Modularity: Ingestion, retrieval, ranking, generation, and UI are isolated layers
- Replaceability: LLM, vector DB, and embedding models must be swappable
- Observability: Every query logs retrieval results, token usage, latency, and confidence score
- Security: Uploaded textbooks are private to the user/session by default
- Scalability: Must support 10k+ pages and concurrent users
- Human-override: Allow human correction of incorrect answers and retraining hooks

## Project Structure

### Documentation (this feature)

```text
specs/2-book-embedded-rag/
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
│   ├── models/
│   │   ├── book.py
│   │   ├── chunk.py
│   │   ├── query_log.py
│   │   └── answer.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   ├── generation_service.py
│   │   └── citation_service.py
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   └── health.py
│   │   └── main.py
│   └── config/
│       ├── settings.py
│       └── database.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

chatbot/
├── src/
│   ├── core/
│   │   ├── rag_engine.js
│   │   ├── mode_resolver.js
│   │   └── citation_formatter.js
│   ├── embed/
│   │   ├── widget.js
│   │   ├── highlight_capture.js
│   │   └── chat_ui.js
│   └── api/
│       └── client.js
└── dist/
    └── book-embedded-rag.js
```

**Structure Decision**: Web application structure with separate backend (FastAPI) and frontend embed (JavaScript) components. The backend handles ingestion, retrieval, and generation logic while the frontend provides the embedded chat widget for digital books.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple components | RAG system requires specialized services for ingestion, retrieval, and generation | Single monolithic service would create tight coupling and reduce modularity |
| Vector + Relational DB | Need both semantic search (vector) and structured logging (relational) | Single database type cannot efficiently serve both semantic search and structured query logging needs |
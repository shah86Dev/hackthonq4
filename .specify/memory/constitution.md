<!-- SYNC IMPACT REPORT:
Version change: 2.1.0 → 2.2.0 (MINOR: Added Ray distributed processing, Dapr sidecars, Qdrant Cloud specifics, and enhanced testing principles)
Modified principles: Scalability Architecture (enhanced with Ray and Qdrant Cloud), Kubernetes-Ready Deployment (enhanced with Dapr), Testing Excellence (enhanced with automated evaluations)
Added principles: Text Selection Support (for JS-based text selection), Deployment Architecture (specific Dapr guidance)
Templates requiring updates: .specify/templates/plan-template.md (✅ updated), .specify/templates/spec-template.md (✅ updated), .specify/templates/tasks-template.md (✅ updated)
Follow-up TODOs: None
-->
# Book-Embedded RAG Chatbot — A Spec-Driven Implementation Constitution

## Core Principles

### Accuracy-First Design
The RAG chatbot MUST restrict responses to book content or user-selected text only, avoiding hallucinations. All responses MUST be grounded in the provided source material with proper citation and context.

### Scalability Architecture
The system MUST use Qdrant Cloud Free Tier for vector storage and Ray for distributed embedding processing to ensure scalability with growing content and user load. The architecture MUST handle large books (>500 pages) without performance degradation and efficiently distribute processing tasks across multiple workers.

### Security-First Integration
The system MUST use API keys for OpenAI, Neon Postgres, and Qdrant. The FastAPI application MUST implement rate limiting to prevent abuse and ensure service availability. All sensitive data MUST be properly secured and validated.

### User Experience Focus
The system MUST support text selection via JS in the book viewer with fallback to full-book retrieval. The embedded chatbot MUST integrate seamlessly into book viewing platforms with minimal disruption to the reading experience.

### Performance Optimization
The system MUST optimize for low-latency queries with <2s response time for 95% of requests. The system MUST handle concurrent users efficiently and maintain consistent performance across different query types and large books (>500 pages).

### Testing Excellence
The system MUST include unit tests for chunking, embedding, retrieval, and generation components. The system MUST use automated evaluations for response relevance and accuracy. All critical paths MUST have comprehensive test coverage with measurable quality metrics.

### Spec-Driven Development
All features must derive from explicit specifications with measurable outcomes. Development follows specification-driven methodology with clear requirements, testable acceptance criteria, and defined success metrics.

### Token-Efficient Architecture
The project uses a token-efficient agent architecture. LLMs must never load MCP tool schemas directly. All heavy logic must execute in scripts. Only skill instructions and final outputs enter context.

### Reproducibility
All implementations must be deterministic, documented, and version-controlled. Developers can reproduce results consistently and build upon proven foundations. All dependencies and configurations MUST be explicitly defined.

## RAG-Specific Technical Principles

### RAG Pipeline Excellence
The system MUST implement a robust RAG pipeline with proper chunking (500-1000 chars with 200-char overlap), embedding using OpenAI models, and semantic retrieval from vector stores. All RAG operations MUST be properly logged and monitored.

### Multi-Modal Content Support
The system MUST support various content formats including PDF, Markdown, and text for book ingestion. Content processing MUST preserve structure and meaning while enabling effective retrieval.

### API-First Design
The system MUST provide well-documented RESTful APIs with clear contracts, proper error handling, and consistent response formats. API endpoints MUST follow standard conventions and include comprehensive documentation.

### Data Integrity and Traceability
All book content, chunks, queries, and responses MUST be properly tracked with metadata. The system MUST maintain complete traceability from source content to generated answers, including citation information. All specs, prompt history, architecture decisions, and tests MUST be versioned Git artifacts for traceability.

## Technical Standards

### Backend Standards
Backend implementation MUST use FastAPI for API layer, Python 3.11+ for implementation, Neon Postgres for relational data, and Qdrant Cloud for vector storage. The system MUST implement proper error handling, logging, and monitoring across all components. Ray MUST be used for distributed embedding processing.

### Frontend Standards
Frontend implementation MUST use React for UI components, support embedding as widgets/iframes, and provide responsive interfaces. The UI MUST support text selection via JavaScript in the book viewer. The UI MUST be accessible and provide clear feedback during API operations.

### Integration Standards
API integration MUST follow RESTful conventions with proper authentication, rate limiting, and error responses. The system MUST handle partial failures gracefully and provide meaningful error messages to users.

### Deployment Architecture
The system MUST be prepared for Kubernetes deployment with Dapr sidecars for agent state management. The architecture MUST leverage Dapr for service discovery, state management, and component orchestration in containerized environments.

## Development Workflow

### Implementation Process
All features must follow the Spec-Driven Development (SDD) methodology. Features require specification, planning, task breakdown, implementation, testing, and review before completion. Quality gates ensure technical accuracy and user value.

### Review and Quality Assurance
Code must undergo technical review by domain experts, security review for potential vulnerabilities, and performance review for efficiency. All changes must pass these quality gates before merging.

## Non-Goals

- **Hallucination-prone responses**: No responses that are not grounded in source material
- **Unscalable architecture**: No systems that cannot handle large books or multiple concurrent users
- **Insecure implementations**: No systems without proper API key management and rate limiting
- **Poor user experience**: No interfaces that lack text selection support or disrupt the reading experience
- **Token-inefficient processing**: No direct MCP tool schema loading in LLMs; all heavy logic must execute in scripts
- **Poor traceability**: No responses without proper citation or source tracking
- **Inadequate testing**: No features without comprehensive test coverage and automated evaluations
- **Non-containerized deployment**: No systems that cannot be deployed to Kubernetes with Dapr sidecars

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, and migration plan. All feature creation and updates must comply with these principles. Developers must verify constitutional compliance during development and review processes.

**Version**: 2.2.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-25
---
id: 003
title: "RAG Chatbot Architecture Plan"
stage: plan
date_iso: "2025-12-25"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "book-rag-chatbot"
branch: "004-book-rag-chatbot"
user: "user"
command: "sp.plan"
labels: ["architecture", "rag-chatbot", "planning", "agents", "qdrant", "openai"]
links:
  spec: "specs/004-book-rag-chatbot/spec.md"
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - specs/004-book-rag-chatbot/plan.md
  - specs/004-book-rag-chatbot/research.md
  - specs/004-book-rag-chatbot/data-model.md
  - specs/004-book-rag-chatbot/contracts/openapi.yaml
  - specs/004-book-rag-chatbot/quickstart.md
tests_yaml: |
  - None
---

# RAG Chatbot Architecture Plan

## User Prompt
Develop a detailed architecture plan for the RAG chatbot:
- Data Flow: Book text -> Chunking (RecursiveCharacterTextSplitter) -> Embeddings (OpenAI API) -> Store in Qdrant (collection 'book_content', Cosine distance, size 1536) and metadata in Neon Postgres.
- Agents: Use OpenAI Agents SDK to create: RetrievalAgent (queries Qdrant/Neon), GenerationAgent (prompts GPT with context), CoordinatorAgent (handles API requests, checks for selected_text).
- Backend: FastAPI app with async endpoints; integrate QdrantClient, psycopg2 for DB, OpenAI client.
- Frontend: Simple Chainlit chat UI for testing; JS snippet for book embedding: document.addEventListener('mouseup', () => { let selected = window.getSelection().toString(); fetch('/chat', {method: 'POST', body: JSON.stringify({question: 'Explain this', selected_text: selected})}); });
- Deployment Stack: Docker compose for local; Kubernetes with Dapr for prod (actors for session state); Ray clusters for batch processing.
- Integration Points: MCP for agent messaging; A2A for async workflows.
- Milestones: 1. Content ingestion script. 2. Retrieval module. 3. Generation module. 4. API server. 5. UI embedding. 6. Tests and evals.
Include diagrams in plan.md using Mermaid syntax for agent interactions and data flow.

## Response Summary
Created a comprehensive architecture plan for the Book-Embedded RAG Chatbot including:

- Detailed implementation plan with technical context and constitution check
- Research document with technology decisions and alternatives considered
- Data model defining Book Chunk, Query Log, and Book Metadata entities
- OpenAPI contract specification for the API endpoints
- Quickstart guide for local setup and deployment

The plan includes Mermaid diagrams for data flow and agent architecture, addresses all constitutional principles, and provides a clear multi-service architecture with backend, frontend, and shared deployment configurations.

## Outcome
Successfully created a detailed architecture plan that addresses all requirements from the user prompt, including the multi-agent architecture, data flow, technology stack, and deployment strategy. The plan is aligned with constitutional principles and includes all required artifacts.

## Evaluation
The plan comprehensively covers all aspects of the requested RAG chatbot architecture while maintaining compliance with constitutional principles. All required artifacts have been created and the architecture supports the specified milestones and implementation approach.
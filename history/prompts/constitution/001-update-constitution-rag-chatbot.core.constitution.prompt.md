---
id: 001
title: "Update Constitution RAG Chatbot"
stage: constitution
date_iso: "2025-12-25"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "main"
user: "user"
command: "sp.constitution"
labels: ["constitution", "rag-chatbot", "scalability", "security", "testing"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - .specify/memory/constitution.md
tests_yaml: |
  - None
---

# Update Constitution RAG Chatbot

## User Prompt
Define core principles for the RAG chatbot project:
- Prioritize accuracy by restricting responses to book content or user-selected text only, avoiding hallucinations.
- Ensure scalability using Qdrant Cloud Free Tier for vector storage and Ray for distributed embedding processing.
- Integrate security: Use API keys for OpenAI, Neon Postgres, and Qdrant; implement rate limiting in FastAPI.
- Focus on user experience: Support text selection via JS in the book viewer, with fallback to full-book retrieval.
- Testing standards: Include unit tests for chunking, embedding, retrieval, and generation; use automated evaluations for response relevance.
- Performance: Optimize for low-latency queries (<2s response time) and handle large books (>500 pages) efficiently.
- Deployment: Prepare for Kubernetes with Dapr sidecars for agent state management.
- Treat all specs, prompt history, architecture decisions, and tests as versioned Git artifacts for traceability

## Response Summary
Updated the project constitution to incorporate all requested principles for the Book-Embedded RAG Chatbot. Key changes include:

- Enhanced Scalability Architecture to include Qdrant Cloud Free Tier and Ray for distributed processing
- Updated Security-First Integration to specify Neon Postgres and updated service providers
- Added User Experience Focus to specifically mention text selection via JS in book viewer
- Enhanced Testing Excellence to include automated evaluations for response relevance
- Added Deployment Architecture section specifically addressing Kubernetes with Dapr sidecars
- Updated Backend Standards to include Neon Postgres and Ray for distributed processing
- Enhanced Frontend Standards to include text selection support
- Updated Non-Goals to reflect new requirements

The constitution version was updated from 2.1.0 to 2.2.0 to reflect these important additions and modifications.

## Outcome
The constitution now properly reflects all the core principles requested for the RAG chatbot project, ensuring proper guidance for future development decisions.

## Evaluation
The updated constitution successfully incorporates all requested principles while maintaining the existing structure and governance framework.
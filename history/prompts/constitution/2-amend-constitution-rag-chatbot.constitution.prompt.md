---
id: 2
title: "amend-constitution-rag-chatbot"
stage: "constitution"
date: "2025-12-20"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "main"
user: "user"
command: "/sp.constitution"
labels: ["governance", "architecture", "rag", "chatbot"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - ".specify/memory/constitution.md"
tests: []
outcome: "Constitution updated to include RAG chatbot principles"
evaluation: "Successfully added RAG chatbot section with 8 new principles and updated version from 1.1.0 to 2.0.0"
---

# Prompt History Record

## PROMPT_TEXT
Define core principles for the RAG chatbot project:
- Prioritize accuracy by restricting responses to book content or user-selected text only, avoiding hallucinations.
- Ensure scalability using Qdrant Cloud Free Tier for vector storage and Ray for distributed embedding processing.
- Integrate security: Use API keys for OpenAI, Neon Postgres, and Qdrant; implement rate limiting in FastAPI.
- Focus on user experience: Support text selection via JS in the book viewer, with fallback to full-book retrieval.
- Testing standards: Include unit tests for chunking, embedding, retrieval, and generation; use automated evaluations for response relevance.
- Performance: Optimize for low-latency queries (<2s response time) and handle large books (>500 pages) efficiently.
- Deployment: Prepare for Kubernetes with Dapr sidecars for agent state management.
- Treat all specs, prompt history, architecture decisions, and tests as versioned Git artifacts for traceability.
/sp.constitution

## RESPONSE_TEXT
I've updated the project constitution to include the new RAG chatbot principles while preserving the existing Physical AI textbook principles. The changes include:

1. Added a new "RAG Chatbot Core Principles" section with 8 specific principles:
   - Accuracy-First Design
   - Scalability Architecture
   - Security-First Integration
   - User Experience Focus
   - Testing Excellence
   - Performance Optimization
   - Kubernetes-Ready Deployment
   - Artifact Traceability

2. Added RAG-specific Technical Standards section

3. Extended the Non-Goals section with RAG-specific items

4. Updated the version from 1.1.0 to 2.0.0 (major version bump due to significant new sections)

5. Created a Sync Impact Report at the top of the file documenting the changes

The constitution now comprehensively covers both the original Physical AI textbook project and the new RAG chatbot functionality with specific, testable principles for each aspect.
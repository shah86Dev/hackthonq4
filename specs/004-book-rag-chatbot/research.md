# Research: Book-Embedded RAG Chatbot

## Overview
This document captures research findings for the Book-Embedded RAG Chatbot implementation. It addresses all technical unknowns and clarifies architectural decisions.

## Decision: Multi-Agent Architecture Implementation
**Rationale**: Using OpenAI Agents SDK as specified in the requirements allows for a clean separation of concerns between retrieval, generation, and coordination responsibilities. This approach provides better maintainability and scalability compared to monolithic implementations.

**Alternatives considered**:
- LangChain agents: More complex for this specific use case
- Custom agent framework: Would require more development time
- Direct API calls: Less maintainable and harder to scale

## Decision: Vector Database Selection (Qdrant)
**Rationale**: Qdrant Cloud was specifically requested and provides excellent performance for semantic search with cosine similarity. It has good Python client support and scales well for the expected use case.

**Alternatives considered**:
- Pinecone: Commercial alternative but less flexibility
- Weaviate: Good option but Qdrant was specified
- FAISS: Self-hosted option but requires more infrastructure management

## Decision: Embedding Model (OpenAI text-embedding-ada-002)
**Rationale**: text-embedding-ada-002 offers a good balance of performance and cost for semantic search. It produces 1536-dimensional vectors as specified in requirements.

**Alternatives considered**:
- text-embedding-3-small: Newer model but less proven in production
- text-embedding-3-large: More capable but more expensive
- Sentence Transformers: Self-hosted option but doesn't match requirements

## Decision: Generation Model (GPT-4o)
**Rationale**: GPT-4o provides the best balance of capability, cost, and performance for generating accurate, context-grounded responses as required.

**Alternatives considered**:
- GPT-4 Turbo: Slightly older but comparable performance
- GPT-3.5 Turbo: Less capable for complex reasoning
- Alternative providers: Doesn't align with OpenAI ecosystem requirements

## Decision: Chunking Strategy (RecursiveCharacterTextSplitter)
**Rationale**: RecursiveCharacterTextSplitter provides reliable text segmentation that respects document structure while meeting the 500-1000 character range with 200-character overlap requirements.

**Alternatives considered**:
- CharacterTextSplitter: Less intelligent segmentation
- TokenTextSplitter: More complex to configure for character-based requirements
- Custom splitter: Would require more development and testing

## Decision: Frontend Integration (JS Widget)
**Rationale**: The JavaScript widget approach with window.getSelection() provides seamless integration with existing book viewers without requiring major changes to hosting platforms.

**Alternatives considered**:
- Full iframe solution: More isolated but less integrated
- Browser extension: More complex deployment and maintenance
- Native integration: Requires changes to each book platform

## Decision: Deployment Strategy (Kubernetes + Dapr)
**Rationale**: Kubernetes with Dapr provides the scalability, reliability, and service mesh capabilities needed for production deployment, with Dapr handling state management as specified.

**Alternatives considered**:
- Docker Swarm: Less feature-rich than Kubernetes
- Serverless: Less control over state management and long-running processes
- Traditional VMs: Less scalable and harder to manage

## Decision: Distributed Processing (Ray)
**Rationale**: Ray provides excellent support for distributed processing of large books, meeting the scalability requirement for books >500 pages.

**Alternatives considered**:
- Celery with Redis: More complex setup for this use case
- Apache Spark: Overkill for embedding tasks
- Custom threading: Less robust than Ray's distributed computing framework
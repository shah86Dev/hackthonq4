# ADR-001: Technology Stack Decision for Physical AI Textbook Platform

## Status
Accepted

## Date
2025-12-16

## Context
The Physical AI & Humanoid Robotics textbook platform requires a technology stack that can support:
- Educational content management with 16 chapters of structured content
- RAG (Retrieval Augmented Generation) capabilities for AI chatbot
- Personalization features for different learning backgrounds
- Multi-language support (English/Urdu)
- Integration with ROS2/Isaac Sim for lab exercises
- Scalability to support 1000+ concurrent users

## Decision
We will use a multi-component architecture with:
- **Frontend**: Docusaurus for documentation website with educational content
- **Backend API**: FastAPI for personalization, authentication, and chatbot services
- **Vector Database**: Qdrant for RAG functionality and semantic search
- **Relational Database**: Neon PostgreSQL for user data and content metadata
- **Authentication**: BetterAuth for secure user management
- **Simulation**: ROS2/Isaac Sim integration for lab exercises

## Alternatives Considered
1. **GitBook + Next.js**: Rejected due to limited backend capabilities for custom AI integration and personalization
2. **Custom React app**: Rejected due to increased development time and reduced SEO benefits compared to Docusaurus
3. **Static site generators**: Rejected due to limited backend capabilities needed for personalization features
4. **Pinecone for vector storage**: Rejected due to vendor lock-in concerns
5. **Auth0 for authentication**: Rejected due to complexity and cost considerations

## Consequences
### Positive
- Docusaurus provides excellent SEO and documentation features out-of-the-box
- FastAPI offers async performance and easy API development with automatic documentation
- Qdrant provides scalable vector search capabilities for RAG system
- Neon PostgreSQL offers serverless scaling and PostgreSQL compatibility
- BetterAuth provides simple authentication with good integration capabilities

### Negative
- Multi-component architecture increases deployment complexity
- Learning curve for team members unfamiliar with FastAPI or Docusaurus
- Additional infrastructure costs for multiple services
- Potential integration challenges between components

## References
- specs/1-physical-ai-textbook/plan.md
- specs/1-physical-ai-textbook/research.md
- specs/1-physical-ai-textbook/data-model.md
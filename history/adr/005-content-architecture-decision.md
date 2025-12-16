# ADR-005: Content Architecture for AI-Native Educational Platform

## Status
Accepted

## Date
2025-12-16

## Context
The Physical AI textbook platform requires a content structure that:
- Supports both Docusaurus documentation and RAG systems simultaneously
- Maintains educational quality and organization
- Enables personalization based on student background
- Integrates with simulation environments for lab exercises
- Scales across 16 chapters with theory, labs, assessments, and projects

## Decision
We will use a structured markdown approach with:
- **Metadata-rich markdown**: Content includes structured metadata for AI indexing
- **Modular organization**: Each chapter contains theory, labs, assessments, projects, glossary, and references
- **Chunk-based structure**: Content is divided into semantically meaningful chunks for RAG
- **Personalization hooks**: Built-in mechanisms for content adaptation based on user profiles
- **Multi-format support**: Content available in both English and Urdu with synchronized updates

## Alternatives Considered
1. **Separate content stores**: Rejected as it would create synchronization challenges between Docusaurus and RAG
2. **Database-only content**: Rejected due to complexity of rendering and version control challenges
3. **Static content only**: Rejected as it wouldn't support personalization requirements
4. **Completely separate systems**: Rejected as it would create maintenance overhead and consistency issues
5. **Single flat structure**: Rejected as it wouldn't support the complex educational requirements

## Consequences
### Positive
- Structured markdown maintains readability and version control benefits
- Metadata enables effective AI indexing for RAG system
- Modular organization supports both documentation and educational needs
- Personalization hooks allow content adaptation without duplicating content
- Multi-format support ensures accessibility for diverse student populations

### Negative
- Complexity of maintaining structured metadata across all content
- Need for tooling to validate content structure and metadata
- Potential for content drift if not properly managed
- Learning curve for content authors to follow structured approach
- Additional processing required to convert content for different systems

## References
- specs/1-physical-ai-textbook/plan.md
- specs/1-physical-ai-textbook/research.md
- specs/1-physical-ai-textbook/data-model.md
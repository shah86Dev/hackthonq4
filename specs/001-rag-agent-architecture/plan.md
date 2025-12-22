# Implementation Plan: RAG-Enabled Agent Architecture

**Branch**: `001-rag-agent-architecture` | **Date**: 2025-12-20 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/001-rag-agent-architecture/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The RAG-enabled agent architecture will implement a system that uses Claude as a planner to route user intents to appropriate skills. These skills will execute via Python scripts (locally) or MCP services, with RAG functionality provided through Qdrant vector search. The system will expose APIs via FastAPI and return minimal results per constitutional requirements for token-efficient processing.

## Technical Context

**Language/Version**: Python 3.11 (for scripts and FastAPI), Claude API (for planning)
**Primary Dependencies**: FastAPI, Qdrant, Python standard library, MCP protocol
**Storage**: Qdrant vector database for RAG functionality, temporary file storage for script execution
**Testing**: pytest for Python scripts, API contract testing for FastAPI endpoints
**Target Platform**: Linux server environment (token-efficient architecture per constitution)
**Project Type**: Single project with modular architecture supporting skills and script execution
**Performance Goals**: <5s response time for 95% of queries, 90% intent routing accuracy, minimal token usage per constitutional requirements
**Constraints**: Token-efficient processing (minimal context injection), MCP tool schema loading externally per constitution, lightweight skill execution
**Scale/Scope**: Single agent system supporting concurrent user queries with minimal resource overhead

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

1. **Token-Efficient Architecture**: ✅
   - LLMs (Claude) will NOT load MCP tool schemas directly
   - All heavy logic will execute in Python scripts
   - Only skill instructions and final outputs will enter context

2. **Systems Thinking**: ✅
   - Full integration of intent planning, skill routing, script execution, and RAG functionality
   - Complete system architecture from user query to response

3. **Spec-Driven Learning**: ✅
   - Following specification-driven development with clear objectives from feature spec

4. **Reproducibility**: ✅
   - All components will be documented and version-controlled
   - Deterministic script execution

5. **Industry Alignment**: ✅
   - Using FastAPI, Qdrant, and Claude API - current industry standards

### Potential Violations and Justifications

- No violations identified. All architectural decisions align with constitutional principles.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-agent-architecture/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
rag_agent/
├── api/                 # FastAPI endpoints
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── query.py     # Query processing endpoint
│       └── skills.py    # Skill management endpoints
├── core/                # Core agent logic
│   ├── __init__.py
│   ├── planner.py       # Claude integration for intent planning
│   ├── router.py        # Intent routing to skills
│   ├── executor.py      # Script execution (local/MCP)
│   └── context_manager.py # Token-efficient context management
├── skills/              # Skill definitions and implementations
│   ├── __init__.py
│   ├── base.py          # Base skill interface
│   ├── registry.py      # Skill registry
│   └── implementations/ # Individual skill implementations
│       ├── rag_search.py # RAG search skill
│       ├── script_runner.py # Script execution skill
│       └── [skill_name].py # Additional skills
├── rag/                 # RAG functionality
│   ├── __init__.py
│   ├── vector_store.py  # Qdrant integration
│   ├── retriever.py     # Document retrieval logic
│   └── embedding.py     # Embedding generation
├── scripts/             # External Python scripts
│   ├── __init__.py
│   ├── command_runner.py # Script execution wrapper
│   └── [script_name].py # Individual scripts
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── test_planner.py
    │   ├── test_router.py
    │   └── test_executor.py
    ├── integration/
    │   ├── test_api.py
    │   └── test_rag_integration.py
    └── contract/
        └── test_api_contracts.py
```

**Structure Decision**: Single project with modular architecture supporting skills and script execution. The structure separates concerns into distinct modules: API layer, core agent logic, skills system, RAG functionality, and external scripts. This follows the token-efficient architecture principle by keeping heavy logic in scripts while maintaining minimal context injection.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [No violations] | [All architectural decisions comply with constitution] | [N/A] |

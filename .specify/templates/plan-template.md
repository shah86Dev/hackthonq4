# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]  
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]  
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]
**Project Type**: [single/web/mobile - determines source structure]  
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]  
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

1. **Accuracy-First Design**:
   - [ ] Responses will be restricted to book content or user-selected text only
   - [ ] All responses will be grounded in provided source material with proper citation
   - [ ] Hallucinations will be avoided through strict context limitations

2. **Scalability Architecture**:
   - [ ] Using Qdrant for vector storage with efficient embedding processing
   - [ ] Designed to handle growing content and user load
   - [ ] Architecture handles large books (>500 pages) without performance degradation

3. **Security-First Integration**:
   - [ ] API keys for OpenAI, PostgreSQL, and Qdrant
   - [ ] FastAPI with rate limiting implementation
   - [ ] Proper authentication and authorization measures

4. **User Experience Focus**:
   - [ ] Support for both full-book and selected-text query modes
   - [ ] Intuitive interfaces that integrate seamlessly into book viewing platforms
   - [ ] Minimal disruption to the reading experience

5. **Performance Optimization**:
   - [ ] <2s response time target for 95% of requests
   - [ ] Efficient handling of concurrent users
   - [ ] Consistent performance across different query types

6. **Spec-Driven Development**:
   - [ ] Features derive from explicit specifications with measurable outcomes
   - [ ] Clear requirements with testable acceptance criteria
   - [ ] Defined success metrics

7. **Token-Efficient Architecture**:
   - [ ] Following constitutional requirement for token-efficient processing
   - [ ] Heavy logic will execute in scripts, not in LLM context
   - [ ] Only skill instructions and final outputs will enter context

8. **Reproducibility**:
   - [ ] All implementations deterministic and version-controlled
   - [ ] Dependencies and configurations explicitly defined
   - [ ] Proper documentation for consistent reproduction

### Potential Violations and Justifications

- [ ] List any potential constitutional violations and justifications here

### Post-Design Constitution Re-check

After implementing the detailed design:

1. **Accuracy-First Design**:
   - [ ] API enforces responses are grounded in book content only
   - [ ] Response model includes proper citation information
   - [ ] Context limitations are strictly enforced

2. **Scalability Architecture**:
   - [ ] Vector store supports required scale
   - [ ] API supports concurrent user handling
   - [ ] Rate limiting prevents resource exhaustion

3. **Security-First Integration**:
   - [ ] API endpoints follow security best practices
   - [ ] Rate limiting implemented to prevent abuse
   - [ ] Sensitive data properly handled in models

4. **User Experience Focus**:
   - [ ] API supports both full-book and selected-text queries
   - [ ] Rich response model with citations and context
   - [ ] Intuitive interfaces for seamless integration

5. **Performance Optimization**:
   - [ ] Response times measured and reported in API
   - [ ] Chunked content enables efficient retrieval
   - [ ] Health check endpoint monitors service performance

6. **Spec-Driven Development**:
   - [ ] API contract defines clear interfaces
   - [ ] Response models include testable metrics
   - [ ] Error handling patterns are standardized

7. **Token-Efficient Architecture**:
   - [ ] API minimizes response sizes with efficient data models
   - [ ] Only necessary context is passed between components
   - [ ] Heavy processing happens in dedicated services, not in API layer

8. **Reproducibility**:
   - [ ] API documented with OpenAPI format
   - [ ] Data models fully specified with validation rules
   - [ ] Implementation follows constitutional requirements

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

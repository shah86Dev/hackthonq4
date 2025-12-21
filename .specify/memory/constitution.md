<!-- SYNC IMPACT REPORT:
Version change: 1.1.0 → 2.0.0 (MAJOR: New RAG chatbot section added with significant new principles)
Added sections: RAG Chatbot Core Principles, RAG Chatbot Technical Standards
Modified sections: Non-Goals (added RAG-specific items)
Templates requiring updates: All templates in .specify/templates/ (⚠ pending manual update)
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics — A Spec-Driven Textbook Constitution

## Core Principles

### Embodied Intelligence First
AI must interact with physical laws, sensors, and actuators. All concepts must have physical grounding and real-world applicability. Students learn AI through its interaction with physical systems rather than in isolation.

### Simulation-to-Reality Continuity
Every concept must work in simulation and transfer to hardware. Students develop skills that bridge the gap between simulation and real-world deployment. All labs must be validated in both environments when possible.

### Systems Thinking
Students understand full robotic stacks, not isolated algorithms. The curriculum emphasizes integration of perception, planning, control, and execution in complete robotic systems.

### Spec-Driven Learning
Each chapter derives from explicit learning specs and measurable outcomes. All content follows specification-driven development with clear objectives, requirements, and success criteria.

### Industry Alignment
ROS 2, Gazebo, NVIDIA Isaac, Jetson, and VLA are mandatory technologies. The curriculum aligns with current industry standards and practices to prepare students for real-world robotics careers.

### Hardware-Aware AI
Resource constraints (edge vs workstation) are explicitly taught. Students understand computational limitations and optimization strategies for different hardware platforms.

### Reproducibility
All examples must be deterministic, documented, and version-controlled. Students can reproduce results consistently and build upon proven foundations.

### Token-Efficient Architecture
The project uses a token-efficient agent architecture. LLMs must never load MCP tool schemas directly. All heavy logic must execute in scripts. Only skill instructions and final outputs enter context.

## RAG Chatbot Core Principles

### Accuracy-First Design
The RAG chatbot MUST restrict responses to book content or user-selected text only, avoiding hallucinations. All responses MUST be grounded in the provided source material with proper citation.

### Scalability Architecture
The system MUST use Qdrant Cloud Free Tier for vector storage and Ray for distributed embedding processing to ensure scalability with growing content and user load.

### Security-First Integration
The system MUST use API keys for OpenAI, Neon Postgres, and Qdrant. The FastAPI application MUST implement rate limiting to prevent abuse and ensure service availability.

### User Experience Focus
The system MUST support text selection via JavaScript in the book viewer, with fallback to full-book retrieval. The interface MUST provide intuitive access to both selected-text and full-book query modes.

### Testing Excellence
The system MUST include unit tests for chunking, embedding, retrieval, and generation components. The system MUST use automated evaluations for response relevance and accuracy.

### Performance Optimization
The system MUST optimize for low-latency queries with <2s response time. The system MUST handle large books (>500 pages) efficiently without degradation in performance.

### Kubernetes-Ready Deployment
The system MUST be prepared for Kubernetes deployment with Dapr sidecars for agent state management, ensuring production-ready scalability and reliability.

### Artifact Traceability
All specs, prompt history, architecture decisions, and tests MUST be versioned Git artifacts for complete traceability and auditability.

## Additional Requirements

### Technical Standards
Technology stack requirements include ROS 2, Gazebo, NVIDIA Isaac, Isaac Sim, Unity, Jetson hardware, RealSense sensors, and Unitree robots. Content must maintain compatibility with these platforms and include appropriate setup guides. The architecture must support token-efficient processing with MCP tool schemas loaded externally and heavy logic executed in dedicated scripts.

### RAG-Specific Technical Standards
RAG implementation MUST use Qdrant for vector storage, OpenAI for generation, Neon Postgres for metadata, and FastAPI for the API layer. The system MUST implement proper error handling and logging for all RAG operations.

### Educational Standards
Content must follow pedagogical best practices with clear learning objectives, hands-on labs, assessment tools, and project-based learning approaches. Each chapter must include practical applications of theoretical concepts.

## Development Workflow

### Content Creation Process
All content must be created following the Spec-Driven Development (SDD) methodology. Chapters require specification, planning, task breakdown, implementation, testing, and review before publication. Quality gates ensure educational value and technical accuracy.

### Review and Quality Assurance
Content must undergo technical review by domain experts, educational review for pedagogical quality, and accessibility review for inclusive design. All chapters must pass these quality gates before publication.

## Non-Goals

- **Purely theoretical AI**: No concepts without physical grounding
- **Black-box robotics frameworks**: No opaque systems without understanding of internals
- **Humanoid hype**: No anthropomorphic focus without kinematics, control, and perception rigor
- **Token-inefficient processing**: No direct MCP tool schema loading in LLMs; all heavy logic must execute in scripts
- **Hallucination-prone responses**: No responses that are not grounded in source material
- **Unscalable architecture**: No systems that cannot handle large books or multiple concurrent users
- **Insecure implementations**: No systems without proper API key management and rate limiting

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, and migration plan. All content creation and updates must comply with these principles. Content creators must verify constitutional compliance during development and review processes.

**Version**: 2.0.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-20
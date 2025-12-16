# Physical AI & Humanoid Robotics Textbook Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-16

## Active Technologies

- Docusaurus (v3.x) - Static site generation for textbook content
- FastAPI (0.104.x) - Backend API framework
- Qdrant (1.7.x) - Vector database for RAG
- Neon (PostgreSQL) - Primary database
- OpenAI API - LLM integration for chatbot
- BetterAuth - Authentication system
- ROS2 Humble Hawksbill - Robotics simulation
- Isaac Sim - NVIDIA robotics simulation
- Gazebo - Robot simulation environment
- Unity - 3D simulation environment
- Python 3.11 - Backend development
- JavaScript/TypeScript - Frontend development
- Markdown - Content authoring

## Project Structure

```text
backend/
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   ├── auth/
│   └── rag/
└── tests/

frontend/
├── docs/                # Docusaurus content (16 chapters)
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

chatbot/
├── src/
│   ├── agents/
│   ├── vector_store/
│   └── integration/
└── tests/

simulation/
├── ros2_ws/            # ROS2 workspace for lab exercises
├── isaac_sim/          # Isaac Sim configurations
└── gazebo/             # Gazebo configurations

specs/1-physical-ai-textbook/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── checklists/
```

## Commands

### Docusaurus Commands
- `npm start` - Start development server
- `npm run build` - Build static site
- `npm run serve` - Serve built site locally

### Backend Commands
- `python -m uvicorn main:app --reload` - Start backend
- `pytest` - Run tests
- `python manage.py migrate` - Run migrations

### Chatbot Commands
- `python app.py` - Start chatbot service
- `python init_vector_store.py` - Initialize vector store

### Content Management
- `npm run validate-content` - Validate content structure
- `npm run build:all` - Build all services

## Code Style

### Python
- Use Black for code formatting
- Follow PEP 8 guidelines
- Type hints required for all function signatures
- Use FastAPI's dependency injection patterns

### JavaScript/TypeScript
- Use Prettier for formatting
- Follow Airbnb style guide
- Use TypeScript for all new code
- Docusaurus plugin patterns for content integration

### Markdown
- Use structured markdown with YAML frontmatter
- Include metadata for AI indexing
- Follow textbook template structure

## Recent Changes

- Feature: Physical AI & Humanoid Robotics Textbook (2025-12-16)
  Added: 16-chapter textbook with ROS2/Isaac Sim labs, RAG chatbot, Urdu translation, personalization
- Feature: AI-Native Content Structure (2025-12-16)
  Added: RAG-optimized content, vector storage, chatbot integration
- Feature: Simulation-First Learning (2025-12-16)
  Added: Isaac Sim/Gazebo lab exercises, ROS2 integration

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
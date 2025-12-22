# Quickstart: RAG-Enabled Agent Architecture

## Overview
This guide provides instructions for setting up and running the RAG-enabled agent architecture.

## Prerequisites

- Python 3.11 or higher
- Qdrant vector database (can run locally or remotely)
- Access to Claude API (Anthropic API key)
- (Optional) MCP-compatible services for remote execution

## Installation

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn qdrant-client python-multipart pydantic
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root:

```env
CLAUDE_API_KEY=your_anthropic_api_key
QDRANT_URL=http://localhost:6333  # or your Qdrant instance URL
QDRANT_API_KEY=your_qdrant_api_key  # if required
MCP_ENDPOINT=http://localhost:8000  # if using MCP services
DEBUG=true  # set to false for production
```

### 2. Skill Configuration
Skills are automatically discovered from the `skills/implementations/` directory. To add a new skill:

1. Create a new Python file in `rag_agent/skills/implementations/`
2. Implement the skill by inheriting from the base Skill class
3. Register the skill in the skill registry

Example skill implementation:
```python
# rag_agent/skills/implementations/example_skill.py
from rag_agent.skills.base import Skill

class ExampleSkill(Skill):
    name = "example_skill"
    description = "An example skill implementation"

    async def execute(self, parameters: dict):
        # Your skill logic here
        return {"result": "example output"}
```

## Running the Agent

### 1. Start Qdrant
If running locally:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT_API_KEY=your-api-key \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 2. Initialize Vector Store
```bash
python -m rag_agent.rag.initialize
```

### 3. Start the API Server
```bash
cd rag_agent
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## API Usage

### Query Endpoint
Send a query to the agent:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "session_id": "session-123"
  }'
```

### Skills Management
List available skills:
```bash
curl http://localhost:8000/skills
```

## Adding Documents to RAG

To add documents to the vector store for RAG functionality:

```bash
curl -X POST http://localhost:8000/rag/documents \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

## Script Execution

The agent supports executing Python scripts both locally and via MCP:

### Local Scripts
Place executable Python scripts in the `rag_agent/scripts/` directory. The agent will execute them with appropriate security measures.

### MCP Services
Configure MCP endpoints in your environment variables to enable remote script execution.

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# API contract tests
pytest tests/contract/
```

## Development

### Adding New Skills
1. Create a new skill class in `rag_agent/skills/implementations/`
2. Implement the required methods
3. Add unit tests in `tests/unit/`
4. Update the skill registry

### Extending RAG Functionality
1. Modify the retriever in `rag_agent/rag/retriever.py`
2. Update embedding logic in `rag_agent/rag/embedding.py`
3. Test with integration tests

## Architecture Notes

- The system follows a token-efficient architecture per constitutional requirements
- Claude only receives skill definitions, not implementations
- Heavy processing happens in external scripts
- Only minimal results are injected back into Claude's context
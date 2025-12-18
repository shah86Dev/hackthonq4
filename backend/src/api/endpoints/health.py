from fastapi import APIRouter
from datetime import datetime
import time
from qdrant_client import QdrantClient
from openai import OpenAI
from src.config.settings import settings


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify system status
    """
    start_time = time.time()

    # Initialize service status
    services_status = {
        "database": {"status": "healthy", "response_time": 0},
        "vector_db": {"status": "healthy", "response_time": 0},
        "llm_provider": {"status": "healthy", "response_time": 0}
    }

    # Check Qdrant (vector database)
    try:
        qdrant_start = time.time()
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        # Simple collection list to verify connection
        qdrant_client.get_collections()
        qdrant_time = time.time() - qdrant_start
        services_status["vector_db"]["response_time"] = round(qdrant_time * 1000, 2)  # Convert to ms
    except Exception as e:
        services_status["vector_db"]["status"] = "unhealthy"
        services_status["vector_db"]["error"] = str(e)

    # Check OpenAI (LLM provider)
    try:
        openai_start = time.time()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        # Simple model list to verify connection
        openai_client.models.list()
        openai_time = time.time() - openai_start
        services_status["llm_provider"]["response_time"] = round(openai_time * 1000, 2)  # Convert to ms
    except Exception as e:
        services_status["llm_provider"]["status"] = "unhealthy"
        services_status["llm_provider"]["error"] = str(e)

    # Overall status
    overall_status = "healthy"
    for service in services_status.values():
        if service["status"] == "unhealthy":
            overall_status = "unhealthy"
            break

    total_time = time.time() - start_time

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": services_status,
        "details": {
            "uptime": "N/A",  # Would require a more persistent uptime tracker
            "active_connections": "N/A",  # Would require monitoring implementation
            "pending_tasks": "N/A"  # Would require task queue monitoring
        },
        "response_time_ms": round(total_time * 1000, 2)
    }
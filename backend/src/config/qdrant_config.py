from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .settings import settings


def get_qdrant_client():
    """
    Create and return a Qdrant client instance
    """
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False  # Set to True in production for better performance
    )
    return client


def ensure_collection_exists(client: QdrantClient, collection_name: str = "book_chunks"):
    """
    Ensure the specified collection exists, create it if it doesn't
    """
    try:
        # Try to get the collection to see if it exists
        client.get_collection(collection_name)
    except:
        # Collection doesn't exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # OpenAI ada-002 embedding size
        )
        print(f"Created Qdrant collection: {collection_name}")


# Initialize the client and ensure collections exist when module is imported
qdrant_client = get_qdrant_client()
ensure_collection_exists(qdrant_client)
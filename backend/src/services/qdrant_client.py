from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class QdrantService:
    """
    Service class for handling Qdrant vector database operations
    """

    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

        # Initialize Qdrant client
        if self.api_key:
            self.client = QdrantClient(
                url=self.host,
                port=self.port,
                api_key=self.api_key,
                https=True if "https" in self.host else False
            )
        else:
            self.client = QdrantClient(host=self.host, port=self.port)

        # Initialize the collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with proper configuration
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with 1536 dimensions for OpenAI embeddings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def store_embedding(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """
        Store a single chunk embedding in Qdrant
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            logger.debug(f"Stored embedding for chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Error storing embedding for chunk {chunk_id}: {e}")
            raise

    def store_embeddings_batch(self, chunk_ids: List[str], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]):
        """
        Store multiple chunk embeddings in Qdrant in a batch
        """
        try:
            points = [
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=metadata
                )
                for chunk_id, embedding, metadata in zip(chunk_ids, embeddings, metadata_list)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.debug(f"Stored {len(chunk_ids)} embeddings in batch")
        except Exception as e:
            logger.error(f"Error storing embeddings batch: {e}")
            raise

    def search_similar(self, query_embedding: List[float], book_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the specified book
        """
        try:
            # Search for similar vectors in the collection
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id",
                            match=models.MatchValue(value=book_id)
                        )
                    ]
                ),
                limit=limit
            )

            # Extract results with metadata
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                results.append(result)

            logger.debug(f"Found {len(results)} similar chunks for book {book_id}")
            return results
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            raise

    def delete_book_chunks(self, book_id: str):
        """
        Delete all chunks associated with a specific book
        """
        try:
            # Delete points matching the book_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="book_id",
                                match=models.MatchValue(value=book_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted all chunks for book {book_id}")
        except Exception as e:
            logger.error(f"Error deleting chunks for book {book_id}: {e}")
            raise

    def get_total_chunks(self, book_id: str) -> int:
        """
        Get the total number of chunks for a specific book
        """
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id",
                            match=models.MatchValue(value=book_id)
                        )
                    ]
                )
            )
            return count_result.count
        except Exception as e:
            logger.error(f"Error counting chunks for book {book_id}: {e}")
            return 0


# Global instance
qdrant_service = QdrantService()
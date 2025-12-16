import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "textbook_content"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings

    async def initialize(self):
        """Initialize the Qdrant client and create collection if it doesn't exist"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)

            # Check if collection exists, if not create it
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Size of all-MiniLM-L6-v2 embeddings
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Connected to existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    async def add_content(self, content_list: List[Dict[str, Any]]):
        """Add content chunks to the vector store with embeddings"""
        try:
            points = []
            for content in content_list:
                # Generate embedding for the content
                embedding = self.encoder.encode(content["content"]).tolist()

                # Create a Qdrant point
                point = models.PointStruct(
                    id=content.get("id", str(uuid.uuid4())),
                    vector=embedding,
                    payload={
                        "content": content["content"],
                        "chapter_id": content["chapter_id"],
                        "chunk_type": content["chunk_type"],
                        "metadata": content.get("metadata", {}),
                        "content_urdu": content.get("content_urdu", "")
                    }
                )
                points.append(point)

            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(points)} content chunks to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding content to vector store: {str(e)}")
            raise

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant content based on the query"""
        try:
            # Generate embedding for the query
            query_embedding = self.encoder.encode(query).tolist()

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )

            # Format results
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "content": hit.payload["content"],
                    "chapter_id": hit.payload["chapter_id"],
                    "chunk_type": hit.payload["chunk_type"],
                    "metadata": hit.payload["metadata"],
                    "content_urdu": hit.payload.get("content_urdu", ""),
                    "score": hit.score
                }
                results.append(result)

            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    async def delete_content(self, content_ids: List[str]):
        """Delete specific content chunks from the vector store"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=content_ids
                )
            )
            logger.info(f"Deleted {len(content_ids)} content chunks from vector store")
            return True

        except Exception as e:
            logger.error(f"Error deleting content from vector store: {str(e)}")
            raise

    async def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific content chunk by its ID"""
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[content_id],
                with_payload=True
            )

            if records:
                record = records[0]
                return {
                    "id": record.id,
                    "content": record.payload["content"],
                    "chapter_id": record.payload["chapter_id"],
                    "chunk_type": record.payload["chunk_type"],
                    "metadata": record.payload["metadata"],
                    "content_urdu": record.payload.get("content_urdu", "")
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving content by ID: {str(e)}")
            raise
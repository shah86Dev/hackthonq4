from typing import List, Optional
from uuid import UUID
from openai import OpenAI
from src.config.settings import settings
from src.services.qdrant_client import qdrant_service
from sqlalchemy.orm import Session
from src.models.chunk import Chunk
import logging

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for the given text using OpenAI
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def retrieve_chunks_full_book(self, book_id: str, query: str, top_k: int = None) -> List[dict]:
        """
        Retrieve relevant chunks from the specified book using vector similarity search
        """
        if top_k is None:
            top_k = settings.top_k_chunks

        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)

            # Search in Qdrant for similar chunks
            results = qdrant_service.search_similar(
                query_embedding=query_embedding,
                book_id=book_id,
                limit=top_k
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "text": result["payload"]["text"],
                    "chapter": result["payload"].get("chapter", "N/A"),
                    "section": result["payload"].get("section", "N/A"),
                    "page_range": result["payload"].get("page_range", "N/A"),
                    "score": result["score"]
                }
                formatted_results.append(formatted_result)

            logger.info(f"Retrieved {len(formatted_results)} chunks for book {book_id}")
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving chunks from Qdrant: {e}")
            raise

    def retrieve_chunks_selected_text(self, selected_text: str, query: str) -> List[dict]:
        """
        Return the selected text as a high-priority chunk for context
        """
        try:
            # Create a chunk from the selected text
            selected_chunk = {
                "id": "selected-text",
                "text": selected_text,
                "chapter": "Selected Text",
                "section": "User Selection",
                "page_range": "N/A",
                "score": 1.0  # Highest priority
            }

            logger.info("Returning selected text as high-priority context")
            return [selected_chunk]
        except Exception as e:
            logger.error(f"Error processing selected text: {e}")
            raise

    def retrieve_chunks_with_selected_context(self, book_id: str, query: str, selected_text: Optional[str] = None, top_k: int = None) -> List[dict]:
        """
        Retrieve chunks from the book, prioritizing selected text if provided
        """
        if top_k is None:
            top_k = settings.top_k_chunks

        results = []

        # If selected text is provided, add it as the highest priority chunk
        if selected_text:
            selected_chunk = {
                "id": "selected-text",
                "text": selected_text,
                "chapter": "Selected Text",
                "section": "User Selection",
                "page_range": "N/A",
                "score": 1.0  # Highest priority
            }
            results.append(selected_chunk)
            # Reduce the number of additional chunks to retrieve
            remaining_k = top_k - 1
        else:
            remaining_k = top_k

        # If we still need more chunks, retrieve from the book
        if remaining_k > 0:
            query_embedding = self.create_embedding(query)
            book_chunks = qdrant_service.search_similar(
                query_embedding=query_embedding,
                book_id=book_id,
                limit=remaining_k
            )

            for chunk in book_chunks:
                formatted_chunk = {
                    "id": chunk["id"],
                    "text": chunk["payload"]["text"],
                    "chapter": chunk["payload"].get("chapter", "N/A"),
                    "section": chunk["payload"].get("section", "N/A"),
                    "page_range": chunk["payload"].get("page_range", "N/A"),
                    "score": chunk["score"]
                }
                results.append(formatted_chunk)

        logger.info(f"Returning {len(results)} total chunks with selected text prioritized")
        return results
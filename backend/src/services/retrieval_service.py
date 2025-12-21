from typing import List, Optional
from uuid import UUID
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from src.config.settings import settings


class RetrievalService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def retrieve_chunks_full_book(self, book_id: UUID, query: str, top_k: int = None) -> List[dict]:
        """
        Retrieve relevant chunks from the specified book using vector similarity search
        """
        if top_k is None:
            top_k = settings.top_k_chunks

        # Generate embedding for the query
        query_embedding = self.openai_client.embeddings.create(
            input=query,
            model=settings.embedding_model
        ).data[0].embedding

        # Create filter to only search within the specified book
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="book_id",
                    match=MatchValue(value=str(book_id))
                )
            ]
        )

        # Perform the search
        search_results = self.qdrant_client.search(
            collection_name="book_chunks",
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k
        )

        # Format results
        retrieved_chunks = []
        for result in search_results:
            chunk_data = {
                "chunk_id": result.payload.get("chunk_id"),
                "text": result.payload.get("text"),
                "chapter": result.payload.get("chapter_id"),  # Using chapter_id from ingestion
                "section": result.payload.get("section"),
                "page_range": result.payload.get("page_range"),
                "similarity_score": result.score
            }
            retrieved_chunks.append(chunk_data)

        return retrieved_chunks

    def retrieve_chunks_selected_text(self, selected_text: str, query: str) -> List[dict]:
        """
        Retrieve relevant chunks using only the selected text as context
        """
        # In selected-text mode, we just return the selected text as the context
        # with a high similarity score since it's the exact text the user selected
        chunk_data = {
            "chunk_id": "selected-text",
            "text": selected_text,
            "chapter": "Selected Text",
            "section": "User Selection",
            "page_range": "N/A",
            "similarity_score": 1.0
        }
        return [chunk_data]
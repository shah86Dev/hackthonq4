from src.services.retrieval_service import RetrievalService
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RetrievalAgent:
    """
    Agent responsible for searching Qdrant for top-5 chunks relevant to the query
    """

    def __init__(self):
        self.retrieval_service = RetrievalService()

    def retrieve_chunks(self, book_id: str, query: str, selected_text: str = None) -> List[Dict]:
        """
        Retrieve relevant chunks from the book based on the query
        If selected_text is provided, prioritize it in the search
        """
        try:
            if selected_text:
                # Use the method that prioritizes selected text
                chunks = self.retrieval_service.retrieve_chunks_with_selected_context(
                    book_id=book_id,
                    query=query,
                    selected_text=selected_text
                )
            else:
                # Retrieve chunks from the full book
                chunks = self.retrieval_service.retrieve_chunks_full_book(
                    book_id=book_id,
                    query=query
                )

            logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
            return chunks
        except Exception as e:
            logger.error(f"Error in RetrievalAgent.retrieve_chunks: {e}")
            raise
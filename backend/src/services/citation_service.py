from typing import List, Dict
from src.config.settings import settings


class CitationService:
    def __init__(self):
        pass

    def format_citations(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Format citations from retrieved chunks
        """
        citations = []
        for chunk in retrieved_chunks:
            citation = {
                "text": chunk.get("text", "")[:100] + "..." if len(chunk.get("text", "")) > 100 else chunk.get("text", ""),
                "chapter": chunk.get("chapter", "N/A"),
                "section": chunk.get("section", "N/A"),
                "page_range": chunk.get("page_range", "N/A"),
                "chunk_id": chunk.get("chunk_id", "N/A"),
                "similarity_score": chunk.get("similarity_score", 0.0)
            }
            citations.append(citation)

        return citations

    def extract_citations_from_answer(self, answer: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Extract and format citations from the generated answer based on the retrieved chunks
        """
        # For now, we'll use all retrieved chunks as citations
        # In a more sophisticated implementation, we might identify which specific
        # chunks were referenced in the answer
        return self.format_citations(retrieved_chunks)
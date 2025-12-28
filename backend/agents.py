"""
Book-Embedded RAG Chatbot - Agent Module

This module defines the multi-agent architecture for the RAG system:
- RetrievalAgent: Searches Qdrant for relevant chunks
- GenerationAgent: Generates answers using OpenAI
- CoordinatorAgent: Orchestrates the interaction between agents
"""

from typing import List, Dict, Optional
import logging
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService
from src.models.query_log import QueryLog

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


class GenerationAgent:
    """
    Agent responsible for generating answers using OpenAI Assistant (GPT-4o)
    """

    def __init__(self):
        self.generation_service = GenerationService()

    def generate_answer(self, question: str, context_chunks: List[Dict], mode: str = "full-book", selected_text: str = None) -> Dict:
        """
        Generate an answer based on the question and provided context chunks
        """
        try:
            result = self.generation_service.generate_answer(
                question=question,
                context_chunks=context_chunks,
                mode=mode,
                selected_text=selected_text
            )

            logger.info(f"Generated answer for question: {question[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error in GenerationAgent.generate_answer: {e}")
            raise


class CoordinatorAgent:
    """
    Agent responsible for coordinating the API logic between Retrieval and Generation agents
    """

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.generation_agent = GenerationAgent()
        self.citation_service = CitationService()

    def process_query(self, db: Session, book_id: str, question: str, selected_text: str = None) -> Dict:
        """
        Process a query by coordinating between retrieval and generation agents
        """
        try:
            # Determine the mode based on whether selected text is provided
            mode = "selected-text" if selected_text else "full-book"

            # Step 1: Retrieve relevant chunks using the retrieval agent
            context_chunks = self.retrieval_agent.retrieve_chunks(
                book_id=book_id,
                query=question,
                selected_text=selected_text
            )

            # Step 2: Generate an answer using the generation agent
            generation_result = self.generation_agent.generate_answer(
                question=question,
                context_chunks=context_chunks,
                mode=mode,
                selected_text=selected_text
            )

            # Step 3: Format citations using the citation service
            citations = self.citation_service.format_citations(context_chunks)

            # Step 4: Log the query for analytics
            query_log = QueryLog(
                book_id=uuid.UUID(book_id),
                mode=mode,
                question=question,
                selected_text=selected_text,
                answer=generation_result["answer_text"],
                latency=generation_result["response_time_ms"] / 1000.0,  # Convert to seconds
                tokens_used=generation_result["tokens_used"],
                confidence_score=generation_result["confidence_score"],
                retrieved_chunk_ids=[uuid.UUID(chunk['id']) for chunk in context_chunks if 'id' in chunk]
            )

            # Add to database session but don't commit yet
            db.add(query_log)

            # Step 5: Format the response
            response = {
                "answer": generation_result["answer_text"],
                "mode": mode,
                "book_id": book_id,
                "question": question,
                "selected_text": selected_text,
                "confidence_score": generation_result["confidence_score"],
                "response_time_ms": generation_result["response_time_ms"],
                "tokens_used": generation_result["tokens_used"],
                "context_chunks_used": generation_result["context_chunks_used"],
                "citations": citations
            }

            logger.info(f"Processed query for book {book_id}, response time: {generation_result['response_time_ms']}ms")
            return response
        except Exception as e:
            logger.error(f"Error in CoordinatorAgent.process_query: {e}")
            raise

    def process_query_with_validation(self, db: Session, book_id: str, question: str, selected_text: str = None) -> Dict:
        """
        Process a query with additional validation and error handling
        """
        start_time = datetime.utcnow()

        try:
            # Validate inputs
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")

            if book_id:
                try:
                    uuid.UUID(book_id)
                except ValueError:
                    raise ValueError(f"Invalid book_id format. Must be a valid UUID: {book_id}")

            # Process the query
            result = self.process_query(db, book_id, question, selected_text)

            # Add timing information
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds

            result['total_processing_time_ms'] = processing_time

            return result

        except Exception as e:
            logger.error(f"Error in CoordinatorAgent.process_query_with_validation: {e}")

            # Log the failed query
            try:
                query_log = QueryLog(
                    book_id=uuid.UUID(book_id) if book_id else None,
                    mode="selected-text" if selected_text else "full-book",
                    question=question,
                    selected_text=selected_text,
                    answer=f"Error: {str(e)}",
                    latency=(datetime.utcnow() - start_time).total_seconds(),
                    tokens_used=0,
                    confidence_score=0.0,
                    retrieved_chunk_ids=[],
                    is_successful=False
                )
                db.add(query_log)
                db.commit()
            except Exception as log_error:
                logger.error(f"Failed to log error query: {log_error}")

            raise


# Global instance for easy access
coordinator_agent = CoordinatorAgent()
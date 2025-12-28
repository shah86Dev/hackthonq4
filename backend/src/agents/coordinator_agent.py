from src.agents.retrieval_agent import RetrievalAgent
from src.agents.generation_agent import GenerationAgent
from typing import Dict, List
import logging
from sqlalchemy.orm import Session
from src.models.query_log import QueryLog, QueryLogCreate

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """
    Agent responsible for coordinating the API logic between Retrieval and Generation agents
    """

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.generation_agent = GenerationAgent()

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

            # Step 3: Log the query for analytics
            query_log = QueryLog(
                book_id=book_id,
                mode=mode,
                question=question,
                selected_text=selected_text,
                answer=generation_result["answer_text"],
                latency=generation_result["response_time_ms"] / 1000.0,  # Convert to seconds
                tokens_used=generation_result["tokens_used"],
                confidence_score=generation_result["confidence_score"],
                retrieved_chunk_ids=[chunk["id"] for chunk in generation_result["context_chunks_used"]]
            )

            # Add to database session but don't commit yet
            db.add(query_log)

            # Step 4: Format the response
            response = {
                "answer": generation_result["answer_text"],
                "mode": mode,
                "book_id": book_id,
                "question": question,
                "selected_text": selected_text,
                "confidence_score": generation_result["confidence_score"],
                "response_time_ms": generation_result["response_time_ms"],
                "tokens_used": generation_result["tokens_used"],
                "context_chunks_used": generation_result["context_chunks_used"]
            }

            logger.info(f"Processed query for book {book_id}, response time: {generation_result['response_time_ms']}ms")
            return response
        except Exception as e:
            logger.error(f"Error in CoordinatorAgent.process_query: {e}")
            raise
from src.services.generation_service import GenerationService
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

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
from typing import List, Dict
from openai import OpenAI
from src.config.settings import settings
import logging
import time
from src.models.query_log import QueryLog

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def generate_answer(self, question: str, context_chunks: List[Dict], mode: str = "full-book", selected_text: str = None) -> Dict:
        """
        Generate an answer based on the question and context chunks using OpenAI
        """
        try:
            # Build the context from the retrieved chunks
            context_text = ""
            chunk_info = []

            for chunk in context_chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text.strip():
                    context_text += f"\n\nSection: {chunk.get('section', 'N/A')}\nPage: {chunk.get('page_range', 'N/A')}\nContent: {chunk_text}"
                    chunk_info.append({
                        "id": chunk.get("id"),
                        "section": chunk.get("section", "N/A"),
                        "page_range": chunk.get("page_range", "N/A"),
                        "score": chunk.get("score", 0.0)
                    })

            # Create the system message with instructions for grounding responses in book content
            system_message = {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions based on book content.
                Your responses must be grounded in the provided book content and you must not hallucinate information.
                If the answer is not available in the provided context, clearly state that the information is not in the book.
                Always cite the relevant sections and pages when providing answers.
                For the selected text mode, prioritize the selected text as the primary context for answering."""
            }

            # Create the user message with the question and context
            user_message_content = f"""
            Question: {question}

            Book Context:
            {context_text}
            """

            if selected_text and mode == "selected-text":
                user_message_content += f"""
                Selected Text Context:
                {selected_text}
                """

            user_message_content += """

            Please provide a detailed answer based only on the information provided in the book content.
            If the answer cannot be found in the provided context, clearly state that the information is not available in the book.
            """

            user_message = {
                "role": "user",
                "content": user_message_content
            }

            # Call OpenAI API to generate the response
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model=settings.generation_model,
                messages=[system_message, user_message],
                max_tokens=settings.max_tokens,
                temperature=0.3,  # Lower temperature for more consistent, factual responses
            )
            end_time = time.time()

            # Extract the generated answer
            generated_answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            response_time_ms = int((end_time - start_time) * 1000)

            # Calculate a basic confidence score based on response quality
            confidence_score = self._calculate_confidence_score(generated_answer, context_chunks)

            logger.info(f"Generated answer with {tokens_used} tokens in {response_time_ms}ms")

            return {
                "answer_text": generated_answer,
                "tokens_used": tokens_used,
                "confidence_score": confidence_score,
                "response_time_ms": response_time_ms,
                "context_chunks_used": chunk_info
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _calculate_confidence_score(self, answer: str, context_chunks: List[Dict]) -> float:
        """
        Calculate a basic confidence score based on how well the answer aligns with the context
        """
        # This is a simplified confidence calculation
        # In a real implementation, you might use more sophisticated methods
        if not context_chunks:
            return 0.1  # Very low confidence if no context

        # Check if the answer mentions citing the book or specific sections
        answer_lower = answer.lower()
        has_citation = any(phrase in answer_lower for phrase in [
            "according to the book", "the book states", "mentioned in",
            "cited in", "found in", "section", "page", "chapter"
        ])

        # Base confidence on number of context chunks and presence of citations
        base_confidence = min(0.7 + (len(context_chunks) * 0.1), 0.95)
        if has_citation:
            base_confidence += 0.1

        return min(base_confidence, 1.0)
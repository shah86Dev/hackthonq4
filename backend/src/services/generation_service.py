from typing import List, Dict
from openai import OpenAI
from src.config.settings import settings


class GenerationService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def generate_answer(self, question: str, context_chunks: List[Dict], mode: str = "full-book") -> Dict:
        """
        Generate an answer based on the question and context chunks
        """
        # Build context from retrieved chunks
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(f"Section: {chunk.get('section', 'N/A')}\n")
            context_parts.append(f"Chapter: {chunk.get('chapter', 'N/A')}\n")
            context_parts.append(f"Page: {chunk.get('page_range', 'N/A')}\n")
            context_parts.append(f"Content: {chunk.get('text', '')}\n")
            context_parts.append("---\n")

        context = "".join(context_parts)

        # Create the system message
        if mode == "selected-text":
            system_message = (
                "You are a book-embedded question answering agent. "
                "You may ONLY answer using the provided selected text context. "
                "If the answer is not explicitly stated in the selected text, say: "
                "'I cannot find this information in the selected text.' "
                "No external knowledge. No assumptions. No paraphrased facts without citation."
            )
        else:
            system_message = (
                "You are a book-embedded question answering agent. "
                "You may ONLY answer using the provided book context. "
                "If the answer is not explicitly stated in the book, say: "
                "'I cannot find this information in the book.' "
                "No external knowledge. No assumptions. No paraphrased facts without citation."
            )

        # Create the user message
        user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.3  # Lower temperature for more consistent, fact-based answers
            )

            answer_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Calculate a basic confidence score based on response characteristics
            confidence_score = self._calculate_confidence_score(answer_text, context_chunks)

            return {
                "answer_text": answer_text,
                "confidence_score": confidence_score,
                "tokens_used": tokens_used
            }

        except Exception as e:
            # Handle cases where the model refuses to answer or other errors occur
            if "cannot find this information" in str(e).lower():
                return {
                    "answer_text": "I cannot find this information in the book.",
                    "confidence_score": 0.0,
                    "tokens_used": 0
                }
            else:
                raise e

    def _calculate_confidence_score(self, answer: str, context_chunks: List[Dict]) -> float:
        """
        Calculate a basic confidence score based on answer characteristics
        """
        # If the answer indicates inability to find information, confidence is 0
        if "cannot find this information" in answer.lower():
            return 0.0

        # Calculate based on how much of the context was used
        # This is a simple heuristic - in practice, you might use more sophisticated methods
        avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks) if context_chunks else 0

        # Basic confidence calculation
        confidence = min(avg_similarity, 1.0)  # Cap at 1.0

        # Adjust based on answer length and other factors
        if len(answer) < 20:  # Very short answers might be less confident
            confidence *= 0.8

        return confidence
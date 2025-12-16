import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import os
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    response_text: str
    source_chunk_ids: List[str]
    is_grounding_valid: bool


class RAGAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sessions = {}  # In production, use a proper session store

    async def process_query(self, query: str, session_id: str, language: str = "en") -> RAGResponse:
        """Process a user query using RAG approach"""
        try:
            # Search for relevant content in the vector store
            search_results = await self.vector_store.search(query, limit=5)

            if not search_results:
                # If no relevant content found, return a response indicating this
                return RAGResponse(
                    response_text="I couldn't find relevant information in the textbook to answer your question. Please try rephrasing or consult the appropriate chapter.",
                    source_chunk_ids=[],
                    is_grounding_valid=False
                )

            # Prepare context from search results
            context = self._prepare_context(search_results, language)

            # Generate response using OpenAI
            response_text = await self._generate_response(query, context)

            # Extract source chunk IDs
            source_chunk_ids = [result["id"] for result in search_results]

            # Validate grounding (check if response is based on provided context)
            is_grounding_valid = self._validate_grounding(response_text, context)

            # Store in session history
            await self._store_in_session(session_id, query, response_text, source_chunk_ids)

            return RAGResponse(
                response_text=response_text,
                source_chunk_ids=source_chunk_ids,
                is_grounding_valid=is_grounding_valid
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _prepare_context(self, search_results: List[Dict[str, Any]], language: str) -> str:
        """Prepare context from search results"""
        context_parts = []

        for result in search_results:
            content = result["content"]
            if language == "ur" and result.get("content_urdu"):
                content = result["content_urdu"]

            context_parts.append(
                f"Chapter {result['chapter_id']}, {result['chunk_type']}: {content}"
            )

        return "\n\n".join(context_parts)

    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI with provided context"""
        try:
            system_prompt = """You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Your responses must be grounded ONLY in the provided textbook content.
            Do not use external knowledge or make up information.
            If the provided context doesn't contain relevant information to answer the question,
            politely indicate that you cannot answer based on the textbook content."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide an answer based only on the context provided."}
            ]

            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _validate_grounding(self, response: str, context: str) -> bool:
        """Validate that the response is grounded in the provided context"""
        # This is a simplified grounding validation
        # In a full implementation, this would use more sophisticated techniques
        response_lower = response.lower()
        context_lower = context.lower()

        # Check if the response contains key concepts from the context
        context_words = set(context_lower.split()[:50])  # Take first 50 words as representative
        response_words = set(response_lower.split())

        # If at least some overlap exists, consider it grounded
        overlap = context_words.intersection(response_words)
        grounding_ratio = len(overlap) / max(1, len(context_words))

        # For now, we'll consider it valid if there's at least 10% overlap
        # This is a simplified check - a full implementation would be more sophisticated
        return grounding_ratio >= 0.1

    async def _store_in_session(self, session_id: str, query: str, response: str, source_ids: List[str]):
        """Store the interaction in session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "timestamp": asyncio.get_event_loop().time(),
            "query": query,
            "response": response,
            "source_ids": source_ids
        })

        # Limit session history to last 10 interactions
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]

    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve session history"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        else:
            raise ValueError(f"Session {session_id} not found")

    async def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def clear_all_sessions(self):
        """Clear all sessions (for maintenance)"""
        self.sessions.clear()
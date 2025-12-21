import uuid
import logging
from typing import List
from uuid import UUID
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from openai import OpenAI
from src.models.book import Book
from src.models.content_chunk import ContentChunk
from src.config.settings import settings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Create the collection if it doesn't exist
        try:
            self.qdrant_client.get_collection("book_chunks")
        except:
            self.qdrant_client.create_collection(
                collection_name="book_chunks",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        """
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if overlap is None:
            overlap = settings.chunk_overlap

        # Use character-based chunking to match the spec requirements (500-1000 chars with 200 char overlap)
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length

            chunk = text[start:end]
            if len(chunk.strip()) > 0:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start by (chunk_size - overlap) to create overlap
            start = min(end, start + chunk_size - overlap)

            if start >= text_length:
                break

        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's text-embedding model
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model=settings.embedding_model
        )
        return response.data[0].embedding

    def ingest_book(self, db: Session, book_id: UUID, title: str, version: str, content: str, metadata: dict = None):
        """
        Ingest a book into the system: parse, chunk, embed, and store
        """
        # Create book record
        book = Book(
            id=book_id,
            title=title,
            version=version,
            metadata=metadata
        )
        db.add(book)
        db.commit()
        db.refresh(book)

        # Chunk the content
        chunks = self.chunk_text(content)

        # Process each chunk
        chunk_records = []
        qdrant_points = []

        # Recalculate start positions for each chunk
        current_start = 0
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{book_id}-{i}"

            # Calculate the actual position in the original text
            actual_start = content.find(chunk_text, current_start)
            if actual_start == -1:
                actual_start = current_start  # fallback if exact match not found
            actual_end = actual_start + len(chunk_text)
            current_start = actual_start + 1  # Move past this chunk for next search

            # Generate embedding
            embedding = self.generate_embedding(chunk_text)

            # Create chunk record for database
            chunk_record = ContentChunk(
                chapter_id=str(book_id),  # Use book_id as chapter_id for book-level chunks
                content=chunk_text,
                embedding=embedding,
                chunk_type="theory",
                chunk_metadata={
                    "book_id": str(book_id),
                    "section": f"Section {i}",
                    "page_range": f"{i*2}-{i*2+1}"
                },
                source_start_pos=actual_start,  # Actual position in source text
                source_end_pos=actual_end  # End position in source text
            )
            chunk_records.append(chunk_record)

            # Prepare point for Qdrant
            qdrant_point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "book_id": str(book_id),
                    "chunk_id": chunk_id,
                    "chapter_id": str(book_id),  # Use book_id as chapter identifier
                    "section": f"Section {i}",
                    "page_range": f"{i*2}-{i*2+1}",
                    "text": chunk_text
                }
            )
            qdrant_points.append(qdrant_point)

        # Add chunks to database
        for chunk_record in chunk_records:
            db.add(chunk_record)
        db.commit()

        # Add vectors to Qdrant
        self.qdrant_client.upsert(
            collection_name="book_chunks",
            points=qdrant_points
        )

        return {
            "status": "success",
            "book_id": str(book_id),
            "chunks_processed": len(chunks),
            "message": f"Successfully ingested book '{title}' with {len(chunks)} chunks"
        }
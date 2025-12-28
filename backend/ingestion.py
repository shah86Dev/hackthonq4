"""
Book-Embedded RAG Chatbot - Content Ingestion Module

This module provides functionality for processing book content (PDF, Markdown)
for RAG functionality using distributed processing with Ray where appropriate.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
import ray
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import PyPDF2
import markdown
from io import BytesIO
import re

from src.config.settings import settings
from src.services.qdrant_client import qdrant_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key)

# Try to initialize Ray for distributed processing
try:
    if not ray.is_initialized():
        ray.init(address=settings.ray_address, ignore_reinit_error=True)
    RAY_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not initialize Ray: {e}")
    RAY_AVAILABLE = False


class BookIngestionService:
    """
    Service class for ingesting books into the RAG system
    """

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.embedding_model = settings.embedding_model

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content using PyPDF2
        """
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            # Clean up the text
            text = self._clean_extracted_text(text)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def extract_text_from_markdown(self, md_content: str) -> str:
        """
        Extract and process text from Markdown content
        """
        try:
            # Convert markdown to HTML, then strip HTML tags to get clean text
            html = markdown.markdown(md_content)
            # Remove HTML tags
            clean_text = re.sub(r'<[^<]+?>', '', html)

            # Clean up the text
            clean_text = self._clean_extracted_text(clean_text)
            return clean_text
        except Exception as e:
            logger.error(f"Error extracting text from Markdown: {e}")
            raise

    def extract_text_from_txt(self, txt_content: str) -> str:
        """
        Extract text from plain text file (minimal processing needed)
        """
        try:
            # Clean up the text
            clean_text = self._clean_extracted_text(txt_content)
            return clean_text
        except Exception as e:
            logger.error(f"Error processing plain text: {e}")
            raise

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing
        """
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def detect_format(self, content: bytes, filename: str = None) -> str:
        """
        Detect the format of the content based on file extension or content
        """
        if filename:
            # Check file extension
            if filename.lower().endswith('.pdf'):
                return 'pdf'
            elif filename.lower().endswith('.md') or filename.lower().endswith('.markdown'):
                return 'markdown'
            elif filename.lower().endswith('.txt'):
                return 'txt'

        # If no filename or extension, try to detect from content
        # For now, default to text if no extension is provided
        return 'text'

    def process_content(self, content: bytes, file_format: str, filename: str = None) -> str:
        """
        Process content based on its format
        """
        if file_format == 'pdf':
            return self.extract_text_from_pdf(content)
        elif file_format == 'markdown':
            # If content is bytes, decode it to string
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return self.extract_text_from_markdown(content)
        elif file_format == 'txt':
            # If content is bytes, decode it to string
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return self.extract_text_from_txt(content)
        else:
            # Assume it's plain text
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap

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
        response = openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    async def process_book_async(self, book_id: str, title: str, content: str, format_type: str) -> Dict[str, Any]:
        """
        Asynchronously process a book: chunk, embed, and store
        """
        logger.info(f"Starting async processing for book {title} (ID: {book_id})")

        # Chunk the content
        chunks = self.chunk_text(content)
        logger.info(f"Text chunked into {len(chunks)} pieces")

        # Generate embeddings in parallel
        embedding_tasks = [asyncio.to_thread(self.generate_embedding, chunk) for chunk in chunks]
        embeddings = await asyncio.gather(*embedding_tasks)

        # Store chunks in Qdrant and database
        chunk_metadata = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{book_id}-{i}"

            # Prepare metadata for this chunk
            metadata = {
                "book_id": book_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                "format": format_type,
                "title": title
            }

            # Store in Qdrant
            qdrant_service.store_embedding(chunk_id, embedding, metadata)

            chunk_metadata.append({
                "id": chunk_id,
                "index": i,
                "text_preview": metadata["text_preview"]
            })

        logger.info(f"Successfully processed and stored {len(chunks)} chunks for book {book_id}")

        return {
            "book_id": book_id,
            "title": title,
            "format": format_type,
            "total_chunks": len(chunks),
            "status": "completed",
            "chunk_details": chunk_metadata
        }

    def process_large_book_with_ray(self, book_id: str, title: str, content: str, format_type: str) -> Dict[str, Any]:
        """
        Process a large book using Ray for distributed embedding generation
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to sequential processing")
            # Fall back to async processing
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.process_book_async(book_id, title, content, format_type)
            )

        logger.info(f"Starting Ray-based processing for large book {title} (ID: {book_id})")

        # Define a Ray remote function for embedding generation
        @ray.remote
        def generate_embedding_ray(text: str) -> List[float]:
            return self.generate_embedding(text)

        # Chunk the content
        chunks = self.chunk_text(content)
        logger.info(f"Text chunked into {len(chunks)} pieces for distributed processing")

        # Submit embedding tasks to Ray
        embedding_futures = [generate_embedding_ray.remote(chunk) for chunk in chunks]

        # Get results
        embeddings = ray.get(embedding_futures)

        # Store chunks in Qdrant and database
        chunk_metadata = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{book_id}-{i}"

            # Prepare metadata for this chunk
            metadata = {
                "book_id": book_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                "format": format_type,
                "title": title
            }

            # Store in Qdrant
            qdrant_service.store_embedding(chunk_id, embedding, metadata)

            chunk_metadata.append({
                "id": chunk_id,
                "index": i,
                "text_preview": metadata["text_preview"]
            })

        logger.info(f"Successfully processed and stored {len(chunks)} chunks for large book {book_id}")

        return {
            "book_id": book_id,
            "title": title,
            "format": format_type,
            "total_chunks": len(chunks),
            "status": "completed",
            "chunk_details": chunk_metadata
        }

    def ingest_book(self, title: str, content: str, format_type: str, book_id: str = None) -> Dict[str, Any]:
        """
        Main method to ingest a book into the RAG system
        """
        if book_id is None:
            book_id = str(uuid.uuid4())

        logger.info(f"Starting ingestion for book: {title}")

        # Determine if we should use distributed processing based on content size
        content_length = len(content)
        if content_length > 50000:  # More than 50k characters, likely a large book
            logger.info(f"Content size ({content_length} chars) exceeds threshold, using distributed processing")
            return self.process_large_book_with_ray(book_id, title, content, format_type)
        else:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.process_book_async(book_id, title, content, format_type)
            )


# Global instance for easy access
ingestion_service = BookIngestionService()


def main():
    """
    Main function for CLI usage of the ingestion service
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Ingest books into the RAG system')
    parser.add_argument('--file', type=str, required=True, help='Path to the book file to ingest')
    parser.add_argument('--title', type=str, required=True, help='Title of the book')
    parser.add_argument('--format', type=str, choices=['pdf', 'markdown', 'txt'],
                       help='Format of the book (auto-detected if not provided)')
    parser.add_argument('--book-id', type=str, help='ID for the book (auto-generated if not provided)')

    args = parser.parse_args()

    # Read the file content
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} does not exist")
        sys.exit(1)

    with open(file_path, 'rb') as f:
        content = f.read()

    # Detect format if not provided
    if args.format is None:
        args.format = ingestion_service.detect_format(content, args.file)
        print(f"Detected format: {args.format}")

    # Process the content
    text_content = ingestion_service.process_content(content, args.format)

    # Ingest the book
    result = ingestion_service.ingest_book(
        title=args.title,
        content=text_content,
        format_type=args.format,
        book_id=args.book_id
    )

    print(f"Ingestion completed successfully!")
    print(f"Book ID: {result['book_id']}")
    print(f"Format: {result['format']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
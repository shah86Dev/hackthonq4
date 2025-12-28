import uuid
import logging
from typing import List
from uuid import UUID
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from openai import OpenAI
import PyPDF2
import markdown
from io import BytesIO
import re
from src.models.book import Book
from src.models.chunk import Chunk
from src.config.settings import settings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Ray for distributed processing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Distributed processing will not be used.")


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

        # Initialize Ray if available
        if RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(address=settings.ray_address, ignore_reinit_error=True)
            except Exception as e:
                logger.warning(f"Could not initialize Ray: {e}")
                RAY_AVAILABLE = False

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
            clean_text = re.sub('<[^<]+?>', '', html)

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

    def _process_chunk(self, chunk_text: str, chunk_idx: int, book_id: UUID, content: str, current_start: int) -> dict:
        """
        Process a single chunk: calculate positions, generate embedding, and prepare database/qdrant records
        """
        # Calculate the actual position in the original text
        actual_start = content.find(chunk_text, current_start)
        if actual_start == -1:
            actual_start = current_start  # fallback if exact match not found
        actual_end = actual_start + len(chunk_text)
        next_start = actual_start + 1  # Move past this chunk for next search

        # Generate embedding
        embedding = self.generate_embedding(chunk_text)

        # Create chunk record for database
        chunk_record = Chunk(
            book_id=book_id,
            text=chunk_text,
            embedding=embedding,  # Store embedding in database as well
            chunk_id=f"{book_id}-{chunk_idx}",
            chapter=f"Chapter {chunk_idx//10 + 1}",  # Simple chapter grouping
            section=f"Section {chunk_idx}",
            page_range=f"{chunk_idx*2}-{chunk_idx*2+1}",
            source_start_pos=actual_start,  # Actual position in source text
            source_end_pos=actual_end  # End position in source text
        )

        # Prepare point for Qdrant
        qdrant_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "book_id": str(book_id),
                "chunk_id": f"{book_id}-{chunk_idx}",
                "chapter": f"Chapter {chunk_idx//10 + 1}",
                "section": f"Section {chunk_idx}",
                "page_range": f"{chunk_idx*2}-{chunk_idx*2+1}",
                "text": chunk_text,
                "file_type": "text"  # This will be updated later with actual file type
            }
        )

        return {
            "chunk_record": chunk_record,
            "qdrant_point": qdrant_point,
            "next_start": next_start
        }

    def _process_chunk_ray(self, chunk_text: str, chunk_idx: int, book_id: UUID, content: str, current_start: int) -> dict:
        """
        Ray remote function to process a single chunk: calculate positions, generate embedding, and prepare database/qdrant records
        """
        return self._process_chunk(chunk_text, chunk_idx, book_id, content, current_start)

    def ingest_book(self, db: Session, book_id: UUID, title: str, version: str, content: str, file_format: str = "text", metadata: dict = None):
        """
        Ingest a book into the system: parse, chunk, embed, and store
        """
        # Create book record
        book = Book(
            id=book_id,
            title=title,
            version=version,
            metadata=metadata,
            file_type=file_format
        )
        db.add(book)
        db.commit()
        db.refresh(book)

        # Chunk the content
        chunks = self.chunk_text(content)

        # Process chunks - use Ray for distributed processing if available and if we have many chunks
        chunk_records = []
        qdrant_points = []

        if RAY_AVAILABLE and len(chunks) > 10:  # Only use Ray for large books
            logger.info(f"Using Ray for distributed processing of {len(chunks)} chunks")

            # Create Ray remote function
            ray_process_chunk = ray.remote(self._process_chunk_ray)

            # Submit tasks for each chunk
            chunk_futures = []
            current_start = 0
            for i, chunk_text in enumerate(chunks):
                future = ray_process_chunk.remote(self, chunk_text, i, book_id, content, current_start)
                chunk_futures.append(future)
                # Note: For proper position tracking, we'd need to process sequentially
                # For now, we'll handle position tracking differently in the distributed version

            # Get results
            results = ray.get(chunk_futures)

            for result in results:
                chunk_records.append(result["chunk_record"])
                qdrant_points.append(result["qdrant_point"])
        else:
            # Process chunks sequentially for smaller books or when Ray is not available
            current_start = 0
            for i, chunk_text in enumerate(chunks):
                result = self._process_chunk(chunk_text, i, book_id, content, current_start)
                chunk_records.append(result["chunk_record"])
                qdrant_points.append(result["qdrant_point"])
                current_start = result["next_start"]

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
            "file_format": file_format,
            "message": f"Successfully ingested {file_format} book '{title}' with {len(chunks)} chunks"
        }
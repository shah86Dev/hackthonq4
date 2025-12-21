import tempfile
import PyPDF2
import markdown
from typing import List, Tuple
from pathlib import Path


class DocumentProcessor:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

        return text

    def extract_text_from_markdown(self, md_path: str) -> str:
        """
        Extract text content from a Markdown file
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Convert markdown to plain text by removing markdown formatting
                # First convert to HTML then strip HTML tags to get plain text
                html = markdown.markdown(content)
                # Remove HTML tags to get plain text
                import re
                plain_text = re.sub('<[^<]+?>', '', html)
                return plain_text
        except Exception as e:
            raise Exception(f"Error processing Markdown: {str(e)}")

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text based on file extension
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension in ['.md', '.markdown']:
            return self.extract_text_from_markdown(file_path)
        else:
            # For other text files, just read as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                # If UTF-8 fails, try with different encoding
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()

    def chunk_text(self, text: str, chunk_size: int = 750, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with specified size and overlap
        Following the spec: 500-1000 char segments with 200 char overlap
        """
        chunks = []

        start = 0
        while start < len(text):
            # Calculate the end position
            end = start + chunk_size

            # If we're near the end of the text, adjust end to not exceed text length
            if end > len(text):
                end = len(text)

            # Extract the chunk
            chunk = text[start:end]

            # Only add non-empty chunks
            if len(chunk.strip()) > 0:
                chunks.append(chunk)

            # Move start by (chunk_size - overlap) to create overlap
            # But make sure we don't go backwards or exceed text length
            start = min(end, start + chunk_size - overlap)

            # If start equals end, we've reached the end of the text
            if start >= len(text):
                break

        return chunks

    def process_document(self, file_path: str, chunk_size: int = 750, overlap: int = 200) -> Tuple[str, List[str]]:
        """
        Process a document: extract text and create chunks
        Following the spec: 500-1000 char segments with 200 char overlap
        Returns: (full_text, list_of_chunks)
        """
        text = self.extract_text_from_file(file_path)
        chunks = self.chunk_text(text, chunk_size, overlap)
        return text, chunks
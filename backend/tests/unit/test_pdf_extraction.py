import pytest
from unittest.mock import Mock, patch, mock_open
from src.services.ingestion_service import IngestionService
import PyPDF2
from io import BytesIO


class TestPDFExtraction:
    """
    Unit tests for PDF text extraction functionality in ingestion_service.py
    """

    def setup_method(self):
        """Setup method for each test"""
        # Mock the OpenAI client and Qdrant client to avoid external dependencies
        with patch('src.services.ingestion_service.OpenAI'), \
             patch('src.services.ingestion_service.QdrantClient'):
            self.ingestion_service = IngestionService()

    def test_extract_text_from_pdf_basic(self):
        """Test basic PDF text extraction"""
        # Create a mock PDF content with some text
        mock_pdf_content = b"""
        %PDF-1.4
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Count 1 /Kids [3 0 R] >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
           /Contents 4 0 R /Resources << >>
        >>
        endobj
        4 0 obj
        << /Length 44 >>
        stream
        BT
        /F1 12 Tf
        72 720 Td
        (This is a test PDF with some content.) Tj
        ET
        endstream
        endobj
        xref
        0 5
        0000000000 65535 f
        0000000010 00000 n
        0000000053 00000 n
        0000000100 00000 n
        0000000170 00000 n
        trailer
        << /Size 5 /Root 1 0 R >>
        startxref
        400
        %%EOF
        """

        # Mock PyPDF2 to return our test content
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create a mock page
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is a test PDF with some content."

            # Create a mock reader that returns our page
            mock_instance = Mock()
            mock_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_instance

            # Extract text from the mock PDF
            extracted_text = self.ingestion_service.extract_text_from_pdf(mock_pdf_content)

            # Verify the extracted text
            assert "test PDF" in extracted_text
            assert "content" in extracted_text

    def test_extract_text_from_pdf_multiple_pages(self):
        """Test PDF text extraction with multiple pages"""
        # Mock PyPDF2 to return multiple pages
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create mock pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Content from page 1."
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Content from page 2."

            # Create a mock reader that returns our pages
            mock_instance = Mock()
            mock_instance.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_instance

            # Create dummy PDF content (not actually used due to mocking)
            dummy_pdf_content = b"dummy pdf content"

            # Extract text from the mock PDF
            extracted_text = self.ingestion_service.extract_text_from_pdf(dummy_pdf_content)

            # Verify the extracted text contains content from both pages
            assert "page 1" in extracted_text
            assert "page 2" in extracted_text

    def test_extract_text_from_pdf_empty(self):
        """Test PDF text extraction with empty PDF"""
        # Mock PyPDF2 to return empty content
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create a mock page with empty content
            mock_page = Mock()
            mock_page.extract_text.return_value = ""

            # Create a mock reader that returns our page
            mock_instance = Mock()
            mock_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_instance

            # Create dummy PDF content
            dummy_pdf_content = b"dummy pdf content"

            # Extract text from the mock PDF
            extracted_text = self.ingestion_service.extract_text_from_pdf(dummy_pdf_content)

            # Verify the extracted text is empty or just whitespace
            assert extracted_text.strip() == ""

    def test_extract_text_from_pdf_with_special_characters(self):
        """Test PDF text extraction with special characters"""
        special_text = "This PDF has special characters: émojis, naïve, résumé, and symbols: @#$%^&*()"

        # Mock PyPDF2 to return text with special characters
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            # Create a mock page
            mock_page = Mock()
            mock_page.extract_text.return_value = special_text

            # Create a mock reader that returns our page
            mock_instance = Mock()
            mock_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_instance

            # Create dummy PDF content
            dummy_pdf_content = b"dummy pdf content"

            # Extract text from the mock PDF
            extracted_text = self.ingestion_service.extract_text_from_pdf(dummy_pdf_content)

            # Verify the extracted text contains special characters
            assert "émojis" in extracted_text
            assert "naïve" in extracted_text
            assert "résumé" in extracted_text

    def test_extract_text_from_pdf_error_handling(self):
        """Test error handling in PDF text extraction"""
        # Mock PyPDF2 to raise an exception
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("PDF parsing error")

            # Create dummy PDF content
            dummy_pdf_content = b"dummy pdf content"

            # Verify that the error is properly handled/raised
            with pytest.raises(Exception):
                self.ingestion_service.extract_text_from_pdf(dummy_pdf_content)

    def test_detect_format_pdf_by_extension(self):
        """Test format detection for PDF files by extension"""
        # Test with PDF filename
        result = self.ingestion_service.detect_format(b"dummy content", "test.pdf")
        assert result == "pdf"

        # Test with uppercase extension
        result = self.ingestion_service.detect_format(b"dummy content", "test.PDF")
        assert result == "pdf"

        # Test with mixed case extension
        result = self.ingestion_service.detect_format(b"dummy content", "test.PdF")
        assert result == "pdf"

    def test_process_pdf_content(self):
        """Test processing PDF content"""
        # Mock PyPDF2 to return test content
        test_content = "This is content from a PDF file."
        with patch('src.services.ingestion_service.PyPDF2.PdfReader') as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = test_content
            mock_instance = Mock()
            mock_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_instance

            # Process the PDF content
            result = self.ingestion_service.process_content(b"dummy pdf", "pdf", "test.pdf")

            # Verify the result
            assert test_content in result
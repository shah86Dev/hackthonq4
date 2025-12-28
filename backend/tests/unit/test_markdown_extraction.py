import pytest
from unittest.mock import Mock, patch
from src.services.ingestion_service import IngestionService


class TestMarkdownExtraction:
    """
    Unit tests for Markdown text extraction functionality in ingestion_service.py
    """

    def setup_method(self):
        """Setup method for each test"""
        # Mock the OpenAI client and Qdrant client to avoid external dependencies
        with patch('src.services.ingestion_service.OpenAI'), \
             patch('src.services.ingestion_service.QdrantClient'):
            self.ingestion_service = IngestionService()

    def test_extract_text_from_markdown_basic(self):
        """Test basic Markdown text extraction"""
        markdown_content = """
# Title
This is a simple markdown document.

## Section 1
Some content here with **bold** and *italic* text.

### Subsection
More content with [a link](http://example.com).
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that the extracted text contains the main content
        assert "Title" in extracted_text
        assert "This is a simple markdown document" in extracted_text
        assert "Section 1" in extracted_text
        assert "Some content here" in extracted_text
        assert "bold" in extracted_text
        assert "italic" in extracted_text
        assert "Subsection" in extracted_text
        assert "More content" in extracted_text

        # Verify that HTML tags are removed
        assert "<h1>" not in extracted_text
        assert "<h2>" not in extracted_text
        assert "<h3>" not in extracted_text
        assert "<strong>" not in extracted_text
        assert "<em>" not in extracted_text

    def test_extract_text_from_markdown_with_code_blocks(self):
        """Test Markdown extraction with code blocks"""
        markdown_content = """
# Code Example
Here is some code:

```python
def hello_world():
    print("Hello, world!")
```

And some more text.
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that content outside code blocks is preserved
        assert "Code Example" in extracted_text
        assert "Here is some code" in extracted_text
        assert "And some more text" in extracted_text

        # Code should be preserved but without markdown formatting
        assert "def hello_world" in extracted_text
        assert "print" in extracted_text

    def test_extract_text_from_markdown_with_lists(self):
        """Test Markdown extraction with lists"""
        markdown_content = """
# List Example
Here are some items:

1. First item
2. Second item
3. Third item

- Unordered item
- Another item
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that list items are preserved
        assert "List Example" in extracted_text
        assert "Here are some items" in extracted_text
        assert "First item" in extracted_text
        assert "Second item" in extracted_text
        assert "Third item" in extracted_text
        assert "Unordered item" in extracted_text
        assert "Another item" in extracted_text

    def test_extract_text_from_markdown_with_special_characters(self):
        """Test Markdown extraction with special characters"""
        markdown_content = """
# Special Characters
This includes: émojis, naïve, résumé, and symbols: @#$%^&*()
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that special characters are preserved
        assert "émojis" in extracted_text
        assert "naïve" in extracted_text
        assert "résumé" in extracted_text
        assert "@" in extracted_text
        assert "#" in extracted_text
        assert "$" in extracted_text

    def test_extract_text_from_markdown_empty(self):
        """Test Markdown extraction with empty content"""
        extracted_text = self.ingestion_service.extract_text_from_markdown("")

        # Verify that empty input results in empty output
        assert extracted_text.strip() == ""

    def test_extract_text_from_markdown_only_headers(self):
        """Test Markdown extraction with only headers"""
        markdown_content = """
# Header 1
## Header 2
### Header 3
#### Header 4
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that headers are preserved
        assert "Header 1" in extracted_text
        assert "Header 2" in extracted_text
        assert "Header 3" in extracted_text
        assert "Header 4" in extracted_text

    def test_extract_text_from_markdown_with_links_and_images(self):
        """Test Markdown extraction with links and images"""
        markdown_content = """
# Links and Images
This document has [a link](http://example.com) and ![an image](image.jpg).

[Reference style link][1]

[1]: http://example.com
"""
        extracted_text = self.ingestion_service.extract_text_from_markdown(markdown_content)

        # Verify that link and image text is preserved
        assert "Links and Images" in extracted_text
        assert "a link" in extracted_text
        assert "an image" in extracted_text
        assert "Reference style link" in extracted_text

    def test_extract_text_from_markdown_error_handling(self):
        """Test error handling in Markdown text extraction"""
        # Mock the markdown module to raise an exception
        with patch('src.services.ingestion_service.markdown') as mock_markdown:
            mock_markdown.markdown.side_effect = Exception("Markdown processing error")

            # Verify that the error is properly handled/raised
            with pytest.raises(Exception):
                self.ingestion_service.extract_text_from_markdown("# Test")

    def test_detect_format_markdown_by_extension(self):
        """Test format detection for Markdown files by extension"""
        # Test with .md extension
        result = self.ingestion_service.detect_format(b"dummy content", "test.md")
        assert result == "markdown"

        # Test with .markdown extension
        result = self.ingestion_service.detect_format(b"dummy content", "test.markdown")
        assert result == "markdown"

        # Test with uppercase extensions
        result = self.ingestion_service.detect_format(b"dummy content", "test.MD")
        assert result == "markdown"

        result = self.ingestion_service.detect_format(b"dummy content", "test.MARKDOWN")
        assert result == "text"  # Not a recognized extension

    def test_process_markdown_content(self):
        """Test processing Markdown content"""
        markdown_content = "# Test\nThis is **bold** content."

        # Process the Markdown content
        result = self.ingestion_service.process_content(markdown_content.encode('utf-8'), "markdown", "test.md")

        # Verify the result
        assert "Test" in result
        assert "bold" in result
        # HTML tags should be removed
        assert "<strong>" not in result

    def test_clean_extracted_text(self):
        """Test the text cleaning functionality"""
        dirty_text = "  This   has   extra   spaces.\n\n\nAnd   newlines.\t\tTabs too.  "
        cleaned_text = self.ingestion_service._clean_extracted_text(dirty_text)

        # Verify that extra whitespace is normalized
        assert "This has extra spaces." in cleaned_text
        assert "And newlines." in cleaned_text
        assert "Tabs too." in cleaned_text

        # Should not have multiple consecutive spaces
        assert "  " not in cleaned_text
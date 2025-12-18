/**
 * Citation Formatter for Book-Embedded Chatbot
 */
class CitationFormatter {
  /**
   * Format citations for display
   * @param {Array} citations - Array of citation objects from the backend
   * @returns {string} Formatted citations as a string
   */
  static formatCitations(citations) {
    if (!citations || citations.length === 0) {
      return '';
    }

    const formattedCitations = citations.map((citation, index) => {
      const parts = [];

      if (citation.chapter && citation.chapter !== 'N/A') {
        parts.push(`Chapter: ${citation.chapter}`);
      }

      if (citation.section && citation.section !== 'N/A') {
        parts.push(`Section: ${citation.section}`);
      }

      if (citation.page_range && citation.page_range !== 'N/A') {
        parts.push(`Page: ${citation.page_range}`);
      }

      // Include a snippet of the text if available
      if (citation.text) {
        const textPreview = citation.text.length > 100
          ? citation.text.substring(0, 100) + '...'
          : citation.text;
        parts.push(`Text: "${textPreview}"`);
      }

      return `[${index + 1}] ${parts.join(', ')}`;
    });

    return `Sources: ${formattedCitations.join('; ')}`;
  }

  /**
   * Format citations as HTML elements
   * @param {Array} citations - Array of citation objects from the backend
   * @returns {HTMLElement} HTML element containing formatted citations
   */
  static formatCitationsAsHtml(citations) {
    const container = document.createElement('div');
    container.className = 'book-rag-citations';

    if (!citations || citations.length === 0) {
      container.innerHTML = '<p class="no-citations">No citations available</p>';
      return container;
    }

    const title = document.createElement('h4');
    title.textContent = 'Sources:';
    title.className = 'citations-title';
    container.appendChild(title);

    const list = document.createElement('ul');
    list.className = 'citations-list';

    citations.forEach((citation, index) => {
      const listItem = document.createElement('li');
      listItem.className = 'citation-item';

      const parts = [];

      if (citation.chapter && citation.chapter !== 'N/A') {
        parts.push(`<strong>Chapter:</strong> ${citation.chapter}`);
      }

      if (citation.section && citation.section !== 'N/A') {
        parts.push(`<strong>Section:</strong> ${citation.section}`);
      }

      if (citation.page_range && citation.page_range !== 'N/A') {
        parts.push(`<strong>Page:</strong> ${citation.page_range}`);
      }

      // Include a snippet of the text if available
      if (citation.text) {
        const textPreview = citation.text.length > 100
          ? citation.text.substring(0, 100) + '...'
          : citation.text;
        parts.push(`<em>Text:</em> "${textPreview}"`);
      }

      listItem.innerHTML = parts.join(', ');
      list.appendChild(listItem);
    });

    container.appendChild(list);
    return container;
  }
}

export default CitationFormatter;
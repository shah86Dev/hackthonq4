/**
 * RAG Engine for Book-Embedded Chatbot
 */
class RagEngine {
  constructor(config) {
    this.config = config || {};
    this.apiClient = null;
  }

  /**
   * Initialize the RAG engine with API client
   * @param {Object} apiClient - API client for backend communication
   */
  initialize(apiClient) {
    this.apiClient = apiClient;
  }

  /**
   * Ask a question using the RAG system
   * @param {string} question - The question to ask
   * @param {string} bookId - The ID of the book to query
   * @param {string} selectedText - Optional selected text for selected-text mode
   * @returns {Promise<Object>} The answer and citations
   */
  async askQuestion(question, bookId, selectedText = null) {
    if (!this.apiClient) {
      throw new Error('RAG Engine not initialized. Call initialize() first.');
    }

    const mode = selectedText ? 'selected-text' : 'full-book';

    const requestBody = {
      book_id: bookId,
      question: question,
      mode: mode
    };

    if (selectedText) {
      requestBody.selected_text = selectedText;
    }

    try {
      const response = await this.apiClient.query(requestBody);
      return response;
    } catch (error) {
      console.error('Error querying the RAG system:', error);
      throw error;
    }
  }
}

export default RagEngine;
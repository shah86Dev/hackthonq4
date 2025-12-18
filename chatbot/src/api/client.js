/**
 * API Client for Book-Embedded RAG Chatbot
 */
class ApiClient {
  /**
   * Initialize the API client
   * @param {string} baseUrl - Base URL of the backend API
   */
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  /**
   * Query the backend API
   * @param {Object} requestBody - Request body for the query
   * @returns {Promise<Object>} Response from the API
   */
  async query(requestBody) {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API query error:', error);
      throw error;
    }
  }

  /**
   * Check health of the backend API
   * @returns {Promise<Object>} Health check response
   */
  async health() {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/health`);

      if (!response.ok) {
        throw new Error(`Health check failed with status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
   * Ingest a book into the system
   * @param {Object} bookData - Book data to ingest
   * @returns {Promise<Object>} Response from the API
   */
  async ingestBook(bookData) {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/ingest/book`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(bookData)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Book ingestion error:', error);
      throw error;
    }
  }
}

export default ApiClient;
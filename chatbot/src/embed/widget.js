import RagEngine from '../core/rag_engine.js';
import ModeResolver from '../core/mode_resolver.js';
import CitationFormatter from '../core/citation_formatter.js';
import ApiClient from '../api/client.js';
import HighlightCapture from './highlight_capture.js';

/**
 * Book-Embedded RAG Chatbot Widget
 */
class BookRagWidget {
  /**
   * Initialize the widget
   * @param {Object} config - Configuration object
   * @param {string} config.containerId - ID of the container element
   * @param {string} config.backendUrl - URL of the backend API
   * @param {string} config.bookId - ID of the book to query
   */
  constructor(config) {
    this.config = config;
    this.container = document.getElementById(config.containerId);
    this.ragEngine = new RagEngine();
    this.apiClient = new ApiClient(config.backendUrl);

    // Initialize the RAG engine with the API client
    this.ragEngine.initialize(this.apiClient);

    this.bookId = config.bookId;
    this.selectedText = null;

    this.highlightCapture = null;

    this.init();
  }

  /**
   * Initialize the widget UI
   */
  init() {
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Create the widget UI
    this.container.innerHTML = `
      <div id="book-rag-container" class="book-rag-widget">
        <div id="book-rag-header" class="book-rag-header">
          <h3>Book Assistant</h3>
          <div id="book-rag-mode-indicator" class="mode-indicator">Mode: Full Book</div>
        </div>
        <div id="book-rag-chat" class="book-rag-chat">
          <div id="book-rag-messages" class="book-rag-messages"></div>
          <div id="book-rag-input" class="book-rag-input">
            <input type="text" id="book-rag-question" placeholder="Ask a question about this book..." />
            <button id="book-rag-submit">Ask</button>
          </div>
        </div>
      </div>
    `;

    // Add CSS styles
    this.addStyles();

    // Initialize highlight capture
    this.highlightCapture = new HighlightCapture(this);

    // Set up event listeners
    this.setupEventListeners();
  }

  /**
   * Add CSS styles for the widget
   */
  addStyles() {
    // Check if styles are already added
    if (document.getElementById('book-rag-styles')) {
      return;
    }

    const style = document.createElement('style');
    style.id = 'book-rag-styles';
    style.textContent = `
      .book-rag-widget {
        font-family: Arial, sans-serif;
        border: 1px solid #ddd;
        border-radius: 8px;
        max-width: 500px;
        margin: 20px 0;
      }

      .book-rag-header {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-bottom: 1px solid #ddd;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .book-rag-header h3 {
        margin: 0;
        font-size: 16px;
      }

      .mode-indicator {
        font-size: 12px;
        background-color: #e9ecef;
        padding: 3px 8px;
        border-radius: 12px;
      }

      .book-rag-chat {
        padding: 15px;
      }

      .book-rag-messages {
        min-height: 150px;
        max-height: 300px;
        overflow-y: auto;
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 4px;
      }

      .message {
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 4px;
      }

      .user-message {
        background-color: #d4edda;
        text-align: right;
      }

      .bot-message {
        background-color: #e2e3e5;
      }

      .citations {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
        padding-top: 5px;
        border-top: 1px solid #ddd;
      }

      .book-rag-input {
        display: flex;
      }

      #book-rag-question {
        flex: 1;
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 4px 0 0 4px;
      }

      #book-rag-submit {
        padding: 8px 15px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 0 4px 4px 0;
        cursor: pointer;
      }

      #book-rag-submit:hover {
        background-color: #0056b3;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    const questionInput = document.getElementById('book-rag-question');
    const submitButton = document.getElementById('book-rag-submit');

    // Submit on button click
    submitButton.addEventListener('click', () => {
      this.submitQuestion();
    });

    // Submit on Enter key
    questionInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.submitQuestion();
      }
    });
  }

  /**
   * Submit a question to the RAG engine
   */
  async submitQuestion() {
    const questionInput = document.getElementById('book-rag-question');
    const question = questionInput.value.trim();

    if (!question) {
      return;
    }

    // Add user message to UI
    this.addMessage(question, 'user');

    // Clear input
    questionInput.value = '';

    try {
      // Get answer from RAG engine
      const response = await this.ragEngine.askQuestion(
        question,
        this.bookId,
        this.selectedText
      );

      // Add bot response to UI
      this.addMessage(response.answer, 'bot', response.citations);
    } catch (error) {
      console.error('Error getting answer:', error);
      this.addMessage(`Error: ${error.message}`, 'bot');
    }
  }

  /**
   * Add a message to the chat UI
   * @param {string} text - Message text
   * @param {string} sender - 'user' or 'bot'
   * @param {Array} citations - Optional citations
   */
  addMessage(text, sender, citations = null) {
    const messagesContainer = document.getElementById('book-rag-messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    messageDiv.innerHTML = text;

    if (citations && citations.length > 0) {
      const citationsDiv = document.createElement('div');
      citationsDiv.className = 'citations';
      citationsDiv.innerHTML = CitationFormatter.formatCitations(citations);
      messageDiv.appendChild(citationsDiv);
    }

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  /**
   * Update the selected text for RAG mode
   * @param {string} text - Selected text
   */
  updateSelectedText(text) {
    this.selectedText = text;

    // Update mode indicator
    const modeIndicator = document.getElementById('book-rag-mode-indicator');
    const mode = ModeResolver.determineMode(text);
    modeIndicator.textContent = `Mode: ${mode === 'selected-text' ? 'Selected Text' : 'Full Book'}`;
  }
}

// Global initialization function
function init(config) {
  return new BookRagWidget(config);
}

// Export for use as a module
export default { init, BookRagWidget };
/**
 * Chat UI Component for Book-Embedded RAG Chatbot
 * This is a more comprehensive UI component that can be used independently or with the widget
 */
class ChatUI {
  /**
   * Initialize the Chat UI component
   * @param {Object} config - Configuration object
   * @param {string} config.containerId - ID of the container element
   * @param {Function} config.onSendMessage - Callback for when a message is sent
   * @param {Function} config.onModeChange - Callback for when the mode changes
   */
  constructor(config) {
    this.config = config;
    this.container = document.getElementById(config.containerId);
    this.onSendMessage = config.onSendMessage || (() => {});
    this.onModeChange = config.onModeChange || (() => {});

    this.init();
  }

  /**
   * Initialize the UI
   */
  init() {
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Create the chat UI
    this.container.innerHTML = `
      <div id="book-chat-container" class="book-chat-container">
        <div id="book-chat-header" class="book-chat-header">
          <h3>Book Assistant</h3>
          <div id="book-chat-mode-indicator" class="mode-indicator">Mode: Full Book</div>
        </div>
        <div id="book-chat-messages" class="book-chat-messages"></div>
        <div id="book-chat-input-container" class="book-chat-input-container">
          <input
            type="text"
            id="book-chat-input"
            class="book-chat-input"
            placeholder="Ask a question about this book..."
          />
          <button id="book-chat-submit" class="book-chat-submit">Send</button>
        </div>
      </div>
    `;

    // Add CSS styles
    this.addStyles();

    // Set up event listeners
    this.setupEventListeners();
  }

  /**
   * Add CSS styles for the chat UI
   */
  addStyles() {
    // Check if styles are already added
    if (document.getElementById('book-chat-styles')) {
      return;
    }

    const style = document.createElement('style');
    style.id = 'book-chat-styles';
    style.textContent = `
      .book-chat-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        max-width: 600px;
        margin: 20px 0;
        display: flex;
        flex-direction: column;
        height: 500px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }

      .book-chat-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 15px 20px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .book-chat-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
      }

      .mode-indicator {
        font-size: 12px;
        background-color: rgba(255, 255, 255, 0.2);
        padding: 4px 10px;
        border-radius: 12px;
      }

      .book-chat-messages {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background-color: #fafafa;
      }

      .message {
        margin-bottom: 15px;
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 80%;
        word-wrap: break-word;
        position: relative;
      }

      .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
      }

      .bot-message {
        background-color: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
      }

      .message-header {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 4px;
      }

      .citations {
        font-size: 12px;
        color: #6c757d;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #eee;
      }

      .book-chat-input-container {
        display: flex;
        padding: 15px;
        background-color: white;
        border-top: 1px solid #e0e0e0;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
      }

      .book-chat-input {
        flex: 1;
        padding: 12px 15px;
        border: 1px solid #ddd;
        border-radius: 24px;
        outline: none;
        font-size: 14px;
      }

      .book-chat-input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
      }

      .book-chat-submit {
        margin-left: 10px;
        padding: 12px 20px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 24px;
        cursor: pointer;
        font-size: 14px;
        transition: background-color 0.2s;
      }

      .book-chat-submit:hover {
        background-color: #0056b3;
      }

      .book-chat-submit:disabled {
        background-color: #6c757d;
        cursor: not-allowed;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    const input = document.getElementById('book-chat-input');
    const submitButton = document.getElementById('book-chat-submit');

    // Submit on button click
    submitButton.addEventListener('click', () => {
      this.sendMessage();
    });

    // Submit on Enter key (without Shift)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent new line in input
        this.sendMessage();
      }
    });
  }

  /**
   * Send the current message
   */
  sendMessage() {
    const input = document.getElementById('book-chat-input');
    const message = input.value.trim();

    if (!message) {
      return;
    }

    // Add user message to UI
    this.addMessage(message, 'user');

    // Clear input
    input.value = '';

    // Call the configured callback
    this.onSendMessage(message);
  }

  /**
   * Add a message to the chat
   * @param {string} text - Message text
   * @param {string} sender - 'user' or 'bot'
   * @param {Array} citations - Optional citations
   * @param {Date} timestamp - Optional timestamp (defaults to now)
   */
  addMessage(text, sender, citations = null, timestamp = new Date()) {
    const messagesContainer = document.getElementById('book-chat-messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    // Add message content
    messageDiv.innerHTML = `<div class="message-content">${this.escapeHtml(text)}</div>`;

    // Add citations if provided
    if (citations && citations.length > 0) {
      const citationsDiv = document.createElement('div');
      citationsDiv.className = 'citations';

      // Format citations
      const formattedCitations = citations.map((citation, index) => {
        const parts = [];
        if (citation.chapter && citation.chapter !== 'N/A') parts.push(`Ch: ${citation.chapter}`);
        if (citation.section && citation.section !== 'N/A') parts.push(`Sec: ${citation.section}`);
        if (citation.page_range && citation.page_range !== 'N/A') parts.push(`Pg: ${citation.page_range}`);
        return `[${index + 1}] ${parts.join(', ')}`;
      }).join('; ');

      citationsDiv.textContent = `Sources: ${formattedCitations}`;
      messageDiv.appendChild(citationsDiv);
    }

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  /**
   * Update the mode indicator
   * @param {string} mode - The current mode ('full-book' or 'selected-text')
   */
  updateModeIndicator(mode) {
    const modeIndicator = document.getElementById('book-chat-mode-indicator');
    if (modeIndicator) {
      modeIndicator.textContent = `Mode: ${mode === 'selected-text' ? 'Selected Text' : 'Full Book'}`;
      this.onModeChange(mode);
    }
  }

  /**
   * Escape HTML to prevent XSS
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

export default ChatUI;
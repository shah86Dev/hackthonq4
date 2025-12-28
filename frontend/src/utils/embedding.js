/**
 * Embedding utilities for Book-Embedded RAG Chatbot
 * Provides functions to embed the chatbot in different web contexts
 */

class BookRagEmbedder {
    constructor(options = {}) {
        this.options = {
            containerId: options.containerId || 'book-rag-chatbot',
            apiUrl: options.apiUrl || 'http://localhost:8000',
            bookId: options.bookId || null,
            title: options.title || 'Book Assistant',
            theme: options.theme || 'light', // 'light' or 'dark'
            initialOpen: options.initialOpen || false,
            ...options
        };

        this.isInitialized = false;
        this.chatbotElement = null;
    }

    /**
     * Initialize the chatbot embed
     */
    init() {
        if (this.isInitialized) {
            console.warn('BookRagEmbedder already initialized');
            return;
        }

        // Create the chatbot container
        this.createChatbotContainer();

        // Load the necessary CSS
        this.loadStyles();

        // Initialize the chatbot component
        this.initializeChatbot();

        this.isInitialized = true;
        console.log('BookRagEmbedder initialized successfully');
    }

    /**
     * Create the chatbot container element
     */
    createChatbotContainer() {
        // Check if container already exists
        let container = document.getElementById(this.options.containerId);

        if (!container) {
            container = document.createElement('div');
            container.id = this.options.containerId;
            container.style.position = 'fixed';
            container.style.bottom = '20px';
            container.style.right = '20px';
            container.style.zIndex = '10000';
            container.style.width = '400px';
            container.style.height = '600px';
            container.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            container.style.borderRadius = '8px';
            container.style.overflow = 'hidden';
            container.style.display = this.options.initialOpen ? 'block' : 'none';

            document.body.appendChild(container);
        }

        this.chatbotElement = container;
    }

    /**
     * Load necessary CSS styles
     */
    loadStyles() {
        // Check if styles are already loaded
        if (document.getElementById('book-rag-styles')) {
            return;
        }

        const styleElement = document.createElement('style');
        styleElement.id = 'book-rag-styles';
        styleElement.textContent = `
            #${this.options.containerId} {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: ${this.options.theme === 'dark' ? '#1f2937' : '#ffffff'};
                color: ${this.options.theme === 'dark' ? '#ffffff' : '#000000'};
            }

            .book-rag-header {
                padding: 16px;
                background: ${this.options.theme === 'dark' ? '#374151' : '#f3f4f6'};
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .book-rag-messages {
                height: calc(100% - 120px);
                overflow-y: auto;
                padding: 16px;
            }

            .book-rag-message {
                margin-bottom: 12px;
                padding: 8px 12px;
                border-radius: 8px;
                max-width: 80%;
            }

            .book-rag-message.user {
                background: ${this.options.theme === 'dark' ? '#3b82f6' : '#3b82f6'};
                color: white;
                margin-left: auto;
            }

            .book-rag-message.assistant {
                background: ${this.options.theme === 'dark' ? '#374151' : '#e5e7eb'};
            }

            .book-rag-input {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 12px;
                background: ${this.options.theme === 'dark' ? '#1f2937' : '#ffffff'};
                border-top: 1px solid ${this.options.theme === 'dark' ? '#374151' : '#e5e7eb'};
            }

            .book-rag-input input {
                width: calc(100% - 60px);
                padding: 8px 12px;
                border: 1px solid ${this.options.theme === 'dark' ? '#374151' : '#d1d5db'};
                border-radius: 4px;
                background: ${this.options.theme === 'dark' ? '#374151' : '#ffffff'};
                color: ${this.options.theme === 'dark' ? '#ffffff' : '#000000'};
            }

            .book-rag-input button {
                width: 48px;
                margin-left: 8px;
                padding: 8px;
                border: none;
                border-radius: 4px;
                background: ${this.options.theme === 'dark' ? '#3b82f6' : '#3b82f6'};
                color: white;
                cursor: pointer;
            }

            .book-rag-float-button {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: ${this.options.theme === 'dark' ? '#3b82f6' : '#3b82f6'};
                color: white;
                border: none;
                font-size: 24px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10001;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }

            .typing-indicator {
                display: flex;
                align-items: center;
            }

            .typing-indicator span {
                width: 8px;
                height: 8px;
                background: currentColor;
                border-radius: 50%;
                margin: 0 2px;
                animation: typing 1.4s infinite ease-in-out;
            }

            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-5px); }
            }
        `;

        document.head.appendChild(styleElement);
    }

    /**
     * Initialize the chatbot functionality
     */
    initializeChatbot() {
        if (!this.chatbotElement) return;

        // Create the chatbot HTML structure
        this.chatbotElement.innerHTML = `
            <div class="book-rag-header">
                <h3>${this.options.title}</h3>
                <button id="book-rag-close" style="background: none; border: none; font-size: 20px; cursor: pointer;">×</button>
            </div>
            <div class="book-rag-messages" id="book-rag-messages">
                <div class="book-rag-message assistant">Hello! I'm your book assistant. Ask me questions about this book.</div>
            </div>
            <div class="book-rag-input">
                <input type="text" id="book-rag-input" placeholder="Ask a question..." />
                <button id="book-rag-send">→</button>
            </div>
        `;

        // Add event listeners
        this.setupEventListeners();

        // Show/hide based on initialOpen option
        this.chatbotElement.style.display = this.options.initialOpen ? 'block' : 'none';

        // Create floating button if chat is initially closed
        if (!this.options.initialOpen) {
            this.createFloatingButton();
        }
    }

    /**
     * Set up event listeners for the chatbot
     */
    setupEventListeners() {
        const sendButton = document.getElementById('book-rag-send');
        const inputElement = document.getElementById('book-rag-input');
        const closeButton = document.getElementById('book-rag-close');
        const messagesContainer = document.getElementById('book-rag-messages');

        // Send message on button click
        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key
        inputElement.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Close chat
        closeButton.addEventListener('click', () => {
            this.chatbotElement.style.display = 'none';
            this.removeFloatingButton();
            this.createFloatingButton();
        });

        // Set up text selection listener for selected text context
        document.addEventListener('mouseup', () => {
            setTimeout(() => {
                const selectedText = window.getSelection().toString().trim();
                if (selectedText && selectedText.length > 10) { // Only consider meaningful selections
                    this.selectedText = selectedText;
                    console.log('Text selected for context:', selectedText.substring(0, 50) + '...');
                }
            }, 0);
        });
    }

    /**
     * Create the floating chat button
     */
    createFloatingButton() {
        if (document.getElementById('book-rag-float-button')) return;

        const floatButton = document.createElement('button');
        floatButton.id = 'book-rag-float-button';
        floatButton.className = 'book-rag-float-button';
        floatButton.innerHTML = '💬';
        floatButton.title = 'Open Book Assistant';

        floatButton.addEventListener('click', () => {
            this.chatbotElement.style.display = 'block';
            floatButton.remove();
        });

        document.body.appendChild(floatButton);
    }

    /**
     * Remove the floating chat button
     */
    removeFloatingButton() {
        const floatButton = document.getElementById('book-rag-float-button');
        if (floatButton) {
            floatButton.remove();
        }
    }

    /**
     * Send a message to the backend
     */
    async sendMessage() {
        const inputElement = document.getElementById('book-rag-input');
        const message = inputElement.value.trim();

        if (!message) return;

        // Add user message to UI
        this.addMessage(message, 'user');
        inputElement.value = '';

        // Show typing indicator
        const typingElement = document.createElement('div');
        typingElement.className = 'book-rag-message assistant';
        typingElement.id = 'typing-indicator';
        typingElement.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        document.getElementById('book-rag-messages').appendChild(typingElement);

        try {
            // Prepare request body
            const requestBody = {
                question: message,
                book_id: this.options.bookId,
                selected_text: this.selectedText || null
            };

            // Send request to backend
            const response = await fetch(`${this.options.apiUrl}/api/v1/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}`);
            }

            const data = await response.json();

            // Remove typing indicator
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }

            // Add assistant response to UI
            this.addMessage(data.response, 'assistant');

            // Clear selected text after sending
            this.selectedText = null;
        } catch (error) {
            console.error('Error sending message:', error);

            // Remove typing indicator
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }

            // Add error message
            this.addMessage(`Sorry, I encountered an error: ${error.message}`, 'assistant');
        }
    }

    /**
     * Add a message to the chat interface
     */
    addMessage(text, sender) {
        const messagesContainer = document.getElementById('book-rag-messages');
        const messageElement = document.createElement('div');
        messageElement.className = `book-rag-message ${sender}`;
        messageElement.textContent = text;

        messagesContainer.appendChild(messageElement);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    /**
     * Update the book ID
     */
    setBookId(bookId) {
        this.options.bookId = bookId;
    }

    /**
     * Get the current book ID
     */
    getBookId() {
        return this.options.bookId;
    }

    /**
     * Destroy the chatbot instance
     */
    destroy() {
        if (this.chatbotElement) {
            this.chatbotElement.remove();
        }
        this.removeFloatingButton();

        const styleElement = document.getElementById('book-rag-styles');
        if (styleElement) {
            styleElement.remove();
        }

        this.isInitialized = false;
        console.log('BookRagEmbedder destroyed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BookRagEmbedder };
} else if (typeof window !== 'undefined') {
    window.BookRagEmbedder = BookRagEmbedder;
}

// Example usage:
// const embedder = new BookRagEmbedder({
//     bookId: 'your-book-id',
//     apiUrl: 'https://your-api-url.com',
//     title: 'My Book Assistant',
//     theme: 'light'
// });
// embedder.init();

export { BookRagEmbedder };
import React, { useState, useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './styles.module.css';

const EmbeddedChatBot = ({ title = "Chapter Assistant", description = "Ask questions about this chapter" }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('embeddedChatSessionId');
    return saved ? JSON.parse(saved) : Date.now().toString();
  });
  const [selectedText, setSelectedText] = useState('');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Set up text selection listener
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, []);

  // Save session ID to localStorage
  useEffect(() => {
    localStorage.setItem('embeddedChatSessionId', JSON.stringify(sessionId));
  }, [sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), content: inputValue, role: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend chat API - use environment variable or default
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputValue,
          selected_text: selectedText || null,  // Send selected text if available
          session_id: sessionId,
          language: 'en',
          book_id: '12345678-1234-5678-1234-567812345678'  // Default book ID
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        content: data.response,
        role: 'assistant',
        sources: data.source_chunks
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error getting response from chatbot:', error);
      const errorMessage = {
        id: Date.now() + 1,
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={styles.embeddedChatContainer}>
      <div className={styles.embeddedChatHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      {selectedText && (
        <div className={styles.selectedTextIndicator}>
          <strong>Selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
          <button
            onClick={() => setSelectedText('')}
            className={styles.clearSelectionBtn}
            title="Clear selection"
          >
            ×
          </button>
        </div>
      )}

      <div className={styles.embeddedChatMessages}>
        {messages.length === 0 ? (
          <div className={styles.embeddedChatWelcome}>
            <h4>Ask me about this chapter!</h4>
            <p>Highlight text in the chapter and ask questions about it, or ask general questions about the content.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.embeddedChatMessage} ${styles[message.role]}`}
            >
              <div className={styles.messageContent}>
                {message.content}
                {message.sources && message.sources.length > 0 && (
                  <div className={styles.messageSources}>
                    <details>
                      <summary>Sources</summary>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>{source.text || source}</li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
              </div>
              <div className={styles.messageRole}>
                {message.role === 'user' ? 'You' : 'Assistant'}
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className={`${styles.embeddedChatMessage} ${styles.assistant}`}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <div className={styles.messageRole}>Assistant</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className={styles.embeddedChatInputForm} onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about this chapter..."
          disabled={isLoading}
          rows={1}
        />
        <button
          type="submit"
          disabled={isLoading || !inputValue.trim()}
          className={styles.sendButton}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

const EmbeddedChatBotWrapper = (props) => {
  return (
    <BrowserOnly>
      {() => <EmbeddedChatBot {...props} />}
    </BrowserOnly>
  );
};

export default EmbeddedChatBotWrapper;
import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import EmbeddedChatBot from '../EmbeddedChatBot';
import styles from './styles.module.css';

const FloatingChatButton = ({ title = "Chapter Assistant", description = "Ask questions about this chapter" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <BrowserOnly>
      {() => (
        <div className={styles.floatingChatContainer}>
          {isOpen ? (
            <div className={styles.chatBox}>
              <div className={styles.chatHeader}>
                <h3>{title}</h3>
                <button
                  onClick={toggleChat}
                  className={styles.closeButton}
                  aria-label="Close chat"
                >
                  ×
                </button>
              </div>
              <div className={styles.chatContent}>
                <EmbeddedChatBot title={title} description={description} />
              </div>
            </div>
          ) : (
            <button
              className={`${styles.floatingButton} ${isHovered ? styles.hovered : ''}`}
              onClick={toggleChat}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              aria-label="Open chat assistant"
            >
              <svg
                className={styles.chatIcon}
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H17.13L16.5 18.75C16.3125 19.25 15.95 19.6562 15.4875 19.8938C15.025 20.1312 14.5125 20.175 14.0625 20.0125C13.6125 19.85 13.25 19.5062 13.0875 19.0875C12.925 18.6688 12.9812 18.2062 13.2375 17.8125L14.22 16H11.5L10.87 17.75C10.6825 18.25 10.3193 18.6562 9.8568 18.8938C9.39426 19.1312 8.8818 19.175 8.4318 19.0125C7.9818 18.85 7.6193 18.5062 7.4568 18.0875C7.2943 17.6688 7.35055 17.2062 7.6068 16.8125L8.58 15H7C6.46957 15 5.96086 14.7893 5.58579 14.4142C5.21071 14.0391 5 13.5304 5 13V5C5 4.46957 5.21071 3.96086 5.58579 3.58579C5.96086 3.21071 6.46957 3 7 3H17C17.5304 3 18.0391 3.21071 18.4142 3.58579C18.7893 3.96086 19 4.46957 19 5V13C19 13.5304 18.7893 14.0391 18.4142 14.4142C18.0391 14.7893 17.5304 15 17 15H21Z"
                  fill="currentColor"
                />
              </svg>
              {isHovered && <span className={styles.tooltip}>Open Chapter Assistant</span>}
            </button>
          )}
        </div>
      )}
    </BrowserOnly>
  );
};

export default FloatingChatButton;
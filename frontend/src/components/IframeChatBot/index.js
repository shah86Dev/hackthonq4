import React, { useState, useEffect } from 'react';

// IframeChatBot component that wraps the chat functionality in an iframe
const IframeChatBot = ({
    bookId,
    apiUrl = 'http://localhost:8000',
    title = 'Book Assistant',
    width = '400px',
    height = '600px',
    theme = 'light',
    initialOpen = false
}) => {
    const [isLoaded, setIsLoaded] = useState(false);
    const [isOpen, setIsOpen] = useState(initialOpen);

    // The URL for the chatbot iframe
    const chatbotUrl = `${apiUrl}/chat-embed?bookId=${bookId}&title=${encodeURIComponent(title)}&theme=${theme}`;

    // Toggle the chatbot visibility
    const toggleChat = () => {
        setIsOpen(!isOpen);
    };

    // Handle iframe load
    const handleIframeLoad = () => {
        setIsLoaded(true);
    };

    // Close chat handler
    const closeChat = () => {
        setIsOpen(false);
    };

    // If initially closed, show only a floating button
    if (!isOpen) {
        return (
            <div
                onClick={toggleChat}
                style={{
                    position: 'fixed',
                    bottom: '20px',
                    right: '20px',
                    zIndex: 1000,
                    cursor: 'pointer'
                }}
            >
                <div
                    style={{
                        width: '60px',
                        height: '60px',
                        borderRadius: '50%',
                        backgroundColor: theme === 'dark' ? '#3b82f6' : '#3b82f6',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '24px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                        transition: 'all 0.3s ease'
                    }}
                >
                    💬
                </div>
            </div>
        );
    }

    return (
        <div style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width,
            height,
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            {/* Chat Header */}
            <div style={{
                padding: '12px 16px',
                backgroundColor: theme === 'dark' ? '#374151' : '#f3f4f6',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <h3 style={{ margin: 0, fontSize: '16px' }}>{title}</h3>
                <button
                    onClick={closeChat}
                    style={{
                        background: 'none',
                        border: 'none',
                        fontSize: '20px',
                        cursor: 'pointer',
                        color: theme === 'dark' ? '#fff' : '#000'
                    }}
                >
                    ×
                </button>
            </div>

            {/* Chat Iframe */}
            <iframe
                src={chatbotUrl}
                title={title}
                width="100%"
                height={`calc(100% - 50px)`} // Account for header height
                style={{ border: 'none' }}
                onLoad={handleIframeLoad}
            />

            {/* Loading indicator */}
            {!isLoaded && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: theme === 'dark' ? '#fff' : '#000'
                }}>
                    Loading...
                </div>
            )}
        </div>
    );
};

// Function to generate an embeddable script for external websites
export const generateEmbedScript = (options = {}) => {
    const {
        bookId = 'default-book',
        apiUrl = 'http://localhost:8000',
        title = 'Book Assistant',
        theme = 'light'
    } = options;

    // Create a script element that can be embedded in other websites
    const scriptContent = `
        (function() {
            // Create container for the chatbot
            const container = document.createElement('div');
            container.id = 'book-rag-chatbot-container';
            container.style.position = 'fixed';
            container.style.bottom = '20px';
            container.style.right = '20px';
            container.style.zIndex = '10000';
            container.style.width = '400px';
            container.style.height = '600px';
            container.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            container.style.borderRadius = '8px';
            container.style.display = 'none'; // Initially hidden

            // Create iframe
            const iframe = document.createElement('iframe');
            iframe.src = '${apiUrl}/chat-embed?bookId=${bookId}&title=${encodeURIComponent(title)}&theme=${theme}';
            iframe.width = '100%';
            iframe.height = 'calc(100% - 50px)';
            iframe.style.border = 'none';

            // Create header
            const header = document.createElement('div');
            header.style.padding = '12px 16px';
            header.style.backgroundColor = '${theme === 'dark' ? '#374151' : '#f3f4f6'}';
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.innerHTML = '<h3 style="margin: 0; font-size: 16px;">${title}</h3><button id="chatbot-close" style="background: none; border: none; font-size: 20px; cursor: pointer;">×</button>';

            // Create floating button
            const floatButton = document.createElement('button');
            floatButton.id = 'chatbot-float';
            floatButton.innerHTML = '💬';
            floatButton.style.position = 'fixed';
            floatButton.style.bottom = '20px';
            floatButton.style.right = '20px';
            floatButton.style.width = '60px';
            floatButton.style.height = '60px';
            floatButton.style.borderRadius = '50%';
            floatButton.style.backgroundColor = '${theme === 'dark' ? '#3b82f6' : '#3b82f6'}';
            floatButton.style.color = 'white';
            floatButton.style.border = 'none';
            floatButton.style.fontSize = '24px';
            floatButton.style.cursor = 'pointer';
            floatButton.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            floatButton.style.zIndex = '10001';
            floatButton.style.display = 'block';

            // Append elements
            container.appendChild(header);
            container.appendChild(iframe);
            document.body.appendChild(container);
            document.body.appendChild(floatButton);

            // Add event listeners
            document.getElementById('chatbot-close').addEventListener('click', function() {
                container.style.display = 'none';
                floatButton.style.display = 'block';
            });

            floatButton.addEventListener('click', function() {
                container.style.display = 'block';
                floatButton.style.display = 'none';
            });

            // Add text selection functionality
            document.addEventListener('mouseup', function() {
                const selectedText = window.getSelection().toString().trim();
                if (selectedText && selectedText.length > 10) {
                    // In a real implementation, we would send this to the iframe
                    // For now, we'll just store it
                    window.bookRagSelectedText = selectedText;
                }
            });
        })();
    `;

    // Return a script tag that can be injected into a page
    return `<script type="text/javascript">(function() { ${scriptContent} })();</script>`;
};

export default IframeChatBot;
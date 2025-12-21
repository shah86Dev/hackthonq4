import React from 'react';
import Layout from '@theme/Layout';
import ChatBot from '../components/ChatBot';
import styles from './chat.module.css';

function ChatPage() {
  return (
    <Layout title="RAG Chatbot" description="Interactive chatbot for Physical AI & Humanoid Robotics textbook">
      <div className={styles.chatPage}>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--12">
              <div className={styles.chatContainer}>
                <h1>Physical AI & Robotics Chatbot</h1>
                <p className={styles.description}>
                  Ask questions about Physical AI, Humanoid Robotics, and concepts from the textbook.
                  This RAG-powered chatbot retrieves relevant information from the course material to provide accurate answers.
                </p>
                <div className={styles.chatBotWrapper}>
                  <ChatBot />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default ChatPage;
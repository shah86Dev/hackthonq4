import BookRAG from './embed/widget.js';
import RagEngine from './core/rag_engine.js';
import ModeResolver from './core/mode_resolver.js';
import CitationFormatter from './core/citation_formatter.js';
import ApiClient from './api/client.js';
import HighlightCapture from './embed/highlight_capture.js';
import ChatUI from './embed/chat_ui.js';

// Export all components
export {
  BookRAG,
  RagEngine,
  ModeResolver,
  CitationFormatter,
  ApiClient,
  HighlightCapture,
  ChatUI
};

// Also make BookRAG available globally when used as a script tag
if (typeof window !== 'undefined') {
  window.BookRAG = BookRAG;
}

export default BookRAG;
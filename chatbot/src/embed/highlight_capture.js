/**
 * Highlight Capture for Book-Embedded RAG Chatbot
 */
class HighlightCapture {
  /**
   * Initialize highlight capture
   * @param {Object} widget - Reference to the widget instance
   */
  constructor(widget) {
    this.widget = widget;
    this.init();
  }

  /**
   * Initialize highlight capture functionality
   */
  init() {
    document.addEventListener('mouseup', this.handleMouseUp.bind(this));
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        this.clearHighlight();
      }
    });
  }

  /**
   * Handle mouse up event to capture selection
   */
  handleMouseUp() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText) {
      // Update the widget with the selected text
      this.widget.updateSelectedText(selectedText);
    } else {
      // If no text is selected, clear the selected text in the widget
      this.widget.updateSelectedText(null);
    }
  }

  /**
   * Clear the current highlight
   */
  clearHighlight() {
    const selection = window.getSelection();
    selection.removeAllRanges();

    // Update the widget to indicate no text is selected
    this.widget.updateSelectedText(null);
  }

  /**
   * Get the currently selected text
   * @returns {string} The currently selected text
   */
  getCurrentSelection() {
    return window.getSelection().toString().trim();
  }

  /**
   * Get the selected text with additional context
   * @returns {Object} Object containing selected text and context information
   */
  getSelectionWithContext() {
    const selection = window.getSelection();
    if (!selection.toString().trim()) {
      return null;
    }

    const range = selection.getRangeAt(0);
    const selectedText = selection.toString().trim();

    // Try to get surrounding context
    let contextBefore = '';
    let contextAfter = '';

    try {
      // Create ranges for context
      const beforeRange = document.createRange();
      beforeRange.setStart(range.startContainer, 0);
      beforeRange.setEnd(range.startContainer, range.startOffset);
      contextBefore = beforeRange.toString().substring(-50); // Last 50 chars before selection

      const afterRange = document.createRange();
      afterRange.setStart(range.endContainer, range.endOffset);
      afterRange.setEnd(range.endContainer, range.endContainer.length || range.endContainer.textContent.length);
      contextAfter = afterRange.toString().substring(0, 50); // First 50 chars after selection
    } catch (e) {
      // If context extraction fails, continue with just the selected text
      console.warn('Could not extract context around selection:', e);
    }

    return {
      text: selectedText,
      contextBefore,
      contextAfter,
      range
    };
  }
}

export default HighlightCapture;
/**
 * Text Selection utilities for Book-Embedded RAG Chatbot
 * Handles text selection in the book viewer and prepares it for chat context
 */

class TextSelectionManager {
    constructor() {
        this.selectedText = null;
        this.selectionRange = null;
        this.callbacks = {
            onTextSelected: [],
            onTextCleared: []
        };

        this.init();
    }

    /**
     * Initialize the text selection manager
     */
    init() {
        // Listen for text selection events
        document.addEventListener('mouseup', this.handleTextSelection.bind(this));
        document.addEventListener('touchend', this.handleTextSelection.bind(this));

        // Listen for keyboard shortcuts to clear selection
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }

    /**
     * Handle text selection events
     */
    handleTextSelection() {
        const selectedText = this.getSelectedText();

        if (selectedText && selectedText.trim().length > 0) {
            this.selectedText = selectedText;
            this.selectionRange = this.getSelectionRange();

            // Notify all registered callbacks
            this.callbacks.onTextSelected.forEach(callback => {
                callback(selectedText, this.selectionRange);
            });
        } else if (this.selectedText) {
            // Text was cleared
            const previousText = this.selectedText;
            this.clearSelection();

            // Notify callbacks that text was cleared
            this.callbacks.onTextCleared.forEach(callback => {
                callback(previousText);
            });
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyDown(event) {
        // Clear selection with Escape key
        if (event.key === 'Escape') {
            this.clearSelection();
        }

        // Clear selection with Ctrl/Cmd + Shift + X
        if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'X') {
            this.clearSelection();
            event.preventDefault();
        }
    }

    /**
     * Get the currently selected text
     */
    getSelectedText() {
        const selection = window.getSelection();
        return selection.toString().trim();
    }

    /**
     * Get the selection range for potential highlighting
     */
    getSelectionRange() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            return selection.getRangeAt(0);
        }
        return null;
    }

    /**
     * Clear the current text selection
     */
    clearSelection() {
        this.selectedText = null;
        this.selectionRange = null;

        // Remove visual selection
        if (window.getSelection) {
            window.getSelection().removeAllRanges();
        } else if (document.selection) {
            document.selection.empty();
        }
    }

    /**
     * Highlight the selected text
     */
    highlightSelection() {
        if (!this.selectionRange) return;

        const highlightSpan = document.createElement('span');
        highlightSpan.style.backgroundColor = 'yellow';
        highlightSpan.style.opacity = '0.5';

        try {
            this.selectionRange.surroundContents(highlightSpan);
        } catch (error) {
            // If surrounding fails, create a new range and highlight
            console.warn('Could not surround contents, using alternative highlighting method:', error);
            this.alternativeHighlight();
        }
    }

    /**
     * Alternative highlighting method when range.surroundContents fails
     */
    alternativeHighlight() {
        if (!this.selectionRange) return;

        const selectedText = this.selectionRange.toString();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = `<span style="background-color: yellow; opacity: 0.5;">${selectedText}</span>`;

        this.selectionRange.deleteContents();
        this.selectionRange.insertNode(tempDiv.firstChild);
    }

    /**
     * Register a callback for when text is selected
     */
    onTextSelected(callback) {
        this.callbacks.onTextSelected.push(callback);
    }

    /**
     * Register a callback for when text selection is cleared
     */
    onTextCleared(callback) {
        this.callbacks.onTextCleared.push(callback);
    }

    /**
     * Get the currently selected text
     */
    getSelectedText() {
        return this.selectedText;
    }

    /**
     * Get selection context including position and surrounding text
     */
    getSelectionContext() {
        if (!this.selectedText) return null;

        const range = this.selectionRange;
        if (!range) return null;

        // Get surrounding context (e.g., 100 characters before and after)
        const surroundingText = this.getSurroundingText(range, 100);

        return {
            text: this.selectedText,
            context: surroundingText,
            position: this.getTextPosition(range),
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Get surrounding text around the selection
     */
    getSurroundingText(range, contextLength = 100) {
        const startContainer = range.startContainer;
        const startOffset = range.startOffset;

        // Get text before selection
        const beforeRange = document.createRange();
        beforeRange.setStart(startContainer, Math.max(0, startOffset - contextLength));
        beforeRange.setEnd(startContainer, startOffset);
        const beforeText = beforeRange.toString();

        // Get text after selection
        const endContainer = range.endContainer;
        const endOffset = range.endOffset;
        const maxLength = endContainer.textContent ? endContainer.textContent.length : 0;

        const afterRange = document.createRange();
        afterRange.setStart(endContainer, endOffset);
        afterRange.setEnd(endContainer, Math.min(maxLength, endOffset + contextLength));
        const afterText = afterRange.toString();

        return {
            before: beforeText,
            after: afterText
        };
    }

    /**
     * Get the position of the selected text
     */
    getTextPosition(range) {
        const rect = range.getBoundingClientRect();
        return {
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height,
            x: rect.x,
            y: rect.y
        };
    }
}

// Create a global instance
const textSelectionManager = new TextSelectionManager();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TextSelectionManager, textSelectionManager };
} else if (typeof window !== 'undefined') {
    window.TextSelectionManager = TextSelectionManager;
    window.textSelectionManager = textSelectionManager;
}

export { TextSelectionManager, textSelectionManager };
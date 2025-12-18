/**
 * Mode Resolver for determining RAG mode based on context
 */
class ModeResolver {
  /**
   * Determine the appropriate RAG mode based on available context
   * @param {string} selectedText - Text selected by the user
   * @returns {string} The RAG mode ('full-book' or 'selected-text')
   */
  static determineMode(selectedText) {
    // If text is selected, use selected-text mode
    if (selectedText && selectedText.trim().length > 0) {
      return 'selected-text';
    }

    // Otherwise, use full-book mode
    return 'full-book';
  }

  /**
   * Validate if the selected text is appropriate for selected-text mode
   * @param {string} selectedText - Text selected by the user
   * @returns {boolean} Whether the selected text is valid for selected-text mode
   */
  static isValidSelectedText(selectedText) {
    if (!selectedText || selectedText.trim().length === 0) {
      return false;
    }

    // Check if selected text is not too short
    if (selectedText.trim().length < 10) {
      return false;
    }

    // Check if selected text is not too long (optional constraint)
    if (selectedText.length > 10000) {
      console.warn('Selected text is very long. Consider selecting a smaller portion.');
    }

    return true;
  }
}

export default ModeResolver;
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import EmbeddedChatBot from './index';

// Mock the fetch API
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () =>
      Promise.resolve({
        response: 'This is a test response from the backend.',
        source_chunks: [
          { text: 'Test source 1', section: 'Section 1', page_range: '1-5' },
          { text: 'Test source 2', section: 'Section 2', page_range: '6-10' }
        ]
      })
  })
);

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock crypto.randomUUID
Object.defineProperty(global.crypto, 'randomUUID', {
  value: jest.fn(() => 'test-uuid-123'),
  writable: true,
});

describe('EmbeddedChatBot Integration', () => {
  beforeEach(() => {
    fetch.mockClear();
    localStorageMock.getItem.mockClear();
    localStorageMock.setItem.mockClear();
  });

  test('renders chatbot with correct title', () => {
    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    expect(screen.getByText(/Test Book Assistant/i)).toBeInTheDocument();
  });

  test('allows user to type and submit a question', async () => {
    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    // Find the input element
    const input = screen.getByPlaceholderText(/Ask a question/i);

    // Type a question
    fireEvent.change(input, { target: { value: 'What is this book about?' } });

    // Submit the form
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Wait for the response
    await waitFor(() => {
      expect(screen.getByText(/This is a test response from the backend./i)).toBeInTheDocument();
    });

    // Verify fetch was called with correct parameters
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/chat',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: expect.stringContaining('What is this book about?')
      })
    );
  });

  test('handles API errors gracefully', async () => {
    // Mock a failed API response
    fetch.mockImplementationOnce(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      })
    );

    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    // Find the input element
    const input = screen.getByPlaceholderText(/Ask a question/i);

    // Type a question
    fireEvent.change(input, { target: { value: 'Test question' } });

    // Submit the form
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Wait for the error response
    await waitFor(() => {
      expect(screen.getByText(/Sorry, I encountered an error/i)).toBeInTheDocument();
    });
  });

  test('preserves session ID in localStorage', () => {
    localStorageMock.getItem.mockReturnValue(JSON.stringify('existing-session-id'));

    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    // Verify that localStorage was accessed
    expect(localStorageMock.getItem).toHaveBeenCalledWith('embeddedChatSessionId');
  });

  test('handles selected text context', async () => {
    // Mock window.getSelection
    Object.defineProperty(window, 'getSelection', {
      value: () => ({
        toString: () => 'This is selected text for context'
      }),
      writable: true,
    });

    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    // Simulate text selection
    fireEvent.mouseUp(document);

    // Wait for the selected text indicator to appear
    await waitFor(() => {
      expect(screen.getByText(/Selected text:/i)).toBeInTheDocument();
    });

    // Verify the selected text is displayed
    expect(screen.getByText(/This is selected text for context/i)).toBeInTheDocument();

    // Find the input element
    const input = screen.getByPlaceholderText(/Ask a question/i);

    // Type a question
    fireEvent.change(input, { target: { value: 'What does this selected text mean?' } });

    // Submit the form
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Wait for the response
    await waitFor(() => {
      expect(screen.getByText(/This is a test response from the backend./i)).toBeInTheDocument();
    });

    // Verify that the fetch call included the selected text
    const fetchCall = fetch.mock.calls[0];
    const requestBody = JSON.parse(fetchCall[1].body);
    expect(requestBody.selected_text).toBe('This is selected text for context');
  });

  test('displays sources for assistant responses', async () => {
    render(
      <EmbeddedChatBot
        title="Test Book Assistant"
        bookId="test-book-id"
      />
    );

    // Find the input element
    const input = screen.getByPlaceholderText(/Ask a question/i);

    // Type a question
    fireEvent.change(input, { target: { value: 'Test question for sources' } });

    // Submit the form
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Wait for the response with sources
    await waitFor(() => {
      expect(screen.getByText(/Sources/i)).toBeInTheDocument();
    });

    // Verify source details are present
    expect(screen.getByText(/Test source 1/i)).toBeInTheDocument();
    expect(screen.getByText(/Test source 2/i)).toBeInTheDocument();
  });
});
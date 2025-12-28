import chainlit as cl
import requests
import uuid
from typing import Dict, List, Optional

# Configuration
BACKEND_URL = "http://localhost:8000"

@cl.on_chat_start
async def start_chat():
    """
    Initialize the chat session
    """
    # Initialize session state
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("book_id", "default-book")  # This should be set based on the book being read
    cl.user_session.set("selected_text", None)  # Initialize selected text context

    welcome_msg = cl.Message(
        content="Hello! I'm your book assistant. You can ask me questions about the book content, and I'll provide answers based on the book's text. What would you like to know?"
    )
    await welcome_msg.send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming messages from the user
    """
    # Get session information
    session_id = cl.user_session.get("session_id")
    book_id = cl.user_session.get("book_id")
    selected_text = cl.user_session.get("selected_text", None)

    # Get the user's question
    question = message.content

    # Prepare the request to the backend
    chat_request = {
        "question": question,
        "book_id": book_id,
        "session_id": session_id
    }

    # If there's selected text context, include it
    if selected_text:
        chat_request["selected_text"] = selected_text
        # Inform the user that selected text context is being used
        context_msg = cl.Message(content=f"Using selected text as context: '{selected_text[:100]}{'...' if len(selected_text) > 100 else ''}'")
        await context_msg.send()

    try:
        # Call the backend API
        response = requests.post(
            f"{BACKEND_URL}/api/v1/chat",
            json=chat_request,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            # Parse the response
            data = response.json()
            answer = data.get("response", "Sorry, I couldn't generate an answer.")
            source_chunks = data.get("source_chunks", [])
            mode = data.get("mode", "full-book")

            # Create and send the response message
            response_msg = cl.Message(content=answer)
            await response_msg.send()

            # Add mode information
            if mode == "selected-text":
                mode_msg = cl.Message(content="*Answer based on selected text context*")
                await mode_msg.send()

            # Add citations if available
            if source_chunks:
                citation_text = "\n\n**Sources:**\n"
                for i, chunk in enumerate(source_chunks[:3]):  # Limit to first 3 sources
                    section = chunk.get("section", "N/A")
                    page_range = chunk.get("page_range", "N/A")
                    text_preview = chunk.get("text", "")[:100] + "..." if len(chunk.get("text", "")) > 100 else chunk.get("text", "")
                    citation_text += f"- Section: {section}, Page(s): {page_range}\n  Preview: {text_preview}\n"

                citation_msg = cl.Message(content=citation_text)
                await citation_msg.send()
        else:
            error_msg = cl.Message(content=f"Error: {response.status_code} - {response.text}")
            await error_msg.send()

    except requests.exceptions.RequestException as e:
        error_msg = cl.Message(content=f"Error connecting to backend: {str(e)}")
        await error_msg.send()
    except Exception as e:
        error_msg = cl.Message(content=f"An unexpected error occurred: {str(e)}")
        await error_msg.send()

@cl.on_settings_update
async def setup_agent(settings):
    """
    Handle settings updates from the user
    """
    # This can be used to handle any settings updates if needed
    pass

# Function to handle selected text (this would be called from the frontend)
def set_selected_text(selected_text: str):
    """
    Set the selected text in the session for context
    """
    cl.user_session.set("selected_text", selected_text)

# Function to clear selected text
def clear_selected_text():
    """
    Clear the selected text context
    """
    cl.user_session.set("selected_text", None)

# Function to update the book context
def set_book_context(book_id: str):
    """
    Update the current book context
    """
    cl.user_session.set("book_id", book_id)

# Function to get current selected text
def get_selected_text() -> str:
    """
    Get the currently selected text from session
    """
    return cl.user_session.get("selected_text", None)
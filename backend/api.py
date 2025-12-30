"""
Vercel-compatible API entry point for the Book-Embedded RAG Chatbot
"""
from src.main import app

# Vercel expects the FastAPI app to be available as 'app'
# This file serves as the entry point for the Python serverless function
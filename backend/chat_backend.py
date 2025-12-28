from fastapi import FastAPI, HTTPException, Depends
from typing import Optional
import uuid
from pydantic import BaseModel
import uvicorn
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.citation_service import CitationService


app = FastAPI(
    title="Book RAG Chatbot API",
    description="Simple chatbot API without external dependencies",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


class ChatRequest(BaseModel):
    question: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None
    language: Optional[str] = "en"
    book_id: str  # Required book ID


class ChatResponse(BaseModel):
    response: str
    source_chunks: list = []
    session_id: str


@app.get("/")
def read_root():
    return {"message": "Book RAG Chatbot API is running!", "status": "success", "service": "chatbot-backend"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Book RAG Chatbot"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the book RAG chatbot
    """
    start_time = __import__('time').time()

    try:
        # Initialize services
        retrieval_service = RetrievalService()
        generation_service = GenerationService()
        citation_service = CitationService()

        # Validate the book_id is a valid UUID
        try:
            book_id = uuid.UUID(request.book_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid book_id format. Must be a valid UUID.")

        # Determine the mode based on whether selected_text is provided
        if request.selected_text:
            # Use selected text mode
            retrieved_chunks = retrieval_service.retrieve_chunks_selected_text(
                selected_text=request.selected_text,
                query=request.question
            )
            mode = "selected-text"
        else:
            # Use full-book RAG mode
            retrieved_chunks = retrieval_service.retrieve_chunks_full_book(
                book_id=book_id,
                query=request.question,
                top_k=5
            )
            mode = "full-book"

        # Generate answer using the retrieved chunks
        generation_result = generation_service.generate_answer(
            question=request.question,
            context_chunks=retrieved_chunks,
            mode=mode
        )
        answer_text = generation_result["answer_text"]

        # Format citations using the citation service
        citations = citation_service.format_citations(retrieved_chunks)
        source_chunks = citations

        # Create response
        session_id = request.session_id or str(uuid.uuid4())
        response = ChatResponse(
            response=answer_text,
            source_chunks=source_chunks,
            session_id=session_id
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
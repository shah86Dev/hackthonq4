from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import logging
from typing import Callable, Awaitable


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """
    Middleware for logging requests and responses
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        start_time = time.time()

        # Log the incoming request
        logger.info(f"Request: {request.method} {request.url}")

        # Create a custom send function to intercept the response
        async def send_with_logging(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                status_code = message["status"]

                # Log the response
                logger.info(
                    f"Response: {status_code} {request.method} {request.url} "
                    f"({process_time:.2f}s)"
                )

                # Add process time to response headers
                headers = dict(message.get("headers", []))
                headers[b"X-Process-Time"] = str(process_time).encode("utf-8")
                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_with_logging)


def add_cors_middleware(app):
    """
    Add CORS middleware to the FastAPI app
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def add_middleware(app):
    """
    Add all required middleware to the application
    """
    # Add CORS middleware first
    add_cors_middleware(app)

    # Add custom logging middleware
    app.add_middleware(LoggingMiddleware)

    return app
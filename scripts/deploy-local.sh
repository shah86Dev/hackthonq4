#!/bin/bash

# Local deployment script for Physical AI RAG Chatbot

set -e

echo "Starting local deployment of Physical AI RAG Chatbot..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop or docker service."
    exit 1
fi

# Navigate to backend directory
cd backend

# Build and start services
echo "Building and starting services..."
docker-compose -f docker/docker-compose.yml up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Run database migrations
echo "Running database migrations..."
docker-compose -f docker/docker-compose.yml exec backend alembic upgrade head

# Run initial tests
echo "Running initial tests..."
docker-compose -f docker/docker-compose.yml exec backend pytest tests/unit/ -v

echo "Local deployment completed successfully!"
echo "API is available at: http://localhost:8000"
echo "Health check: http://localhost:8000/api/v1/health"
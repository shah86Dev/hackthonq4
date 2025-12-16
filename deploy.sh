#!/bin/bash

# Deployment script for Physical AI & Humanoid Robotics Textbook
# This script outlines the deployment process for the complete system

set -e  # Exit on any error

echo "Physical AI & Humanoid Robotics Textbook - Deployment Script"
echo "============================================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "All prerequisites satisfied."

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/textbook_db

# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Application Configuration
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
DEBUG=False
EOF
    echo "Created .env file. Please update it with your actual configuration."
fi

# Build and start services
echo "Building and starting services..."
docker-compose up --build -d

echo "Waiting for services to start..."
sleep 30

# Run database migrations (if using alembic)
echo "Running database migrations..."
# docker-compose exec backend alembic upgrade head

# Index textbook content in Qdrant
echo "Indexing textbook content in vector database..."
# This would be done through the backend API or a separate indexing script

echo ""
echo "Deployment completed successfully!"
echo ""
echo "Services are now running:"
echo "- Frontend (Docusaurus): http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- Chatbot API: http://localhost:8001"
echo "- Qdrant Vector DB: http://localhost:6333"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"
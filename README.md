# Physical AI & Humanoid Robotics Textbook

A comprehensive university textbook on Physical AI and Humanoid Robotics with AI-powered features, simulation integration, and multilingual support.

## Overview

This project implements a complete educational platform for Physical AI & Humanoid Robotics, featuring:

- **16 Comprehensive Chapters**: Covering ROS2, Gazebo, Isaac Sim, and VLA Models
- **AI-Powered Chatbot**: RAG-based system for textbook Q&A
- **Multilingual Support**: English and Urdu translations
- **Personalization Engine**: Adaptive content based on user background
- **Simulation Integration**: ROS2, Gazebo, and Isaac Sim labs
- **Instructor Resources**: Slides, assessments, and lab guides

## Architecture

The system consists of multiple interconnected services:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   Chatbot       │
│  (Docusaurus)   │───▶│    (FastAPI)     │───▶│   (FastAPI)     │
│                 │    │                  │    │                 │
│  - Textbook UI  │    │  - User Auth     │    │  - RAG System   │
│  - Navigation   │    │  - Content API   │    │  - Vector DB   │
│  - Labs         │    │  - Personalization│    │  - OpenAI      │
└─────────────────┘    │  - Translation   │    └─────────────────┘
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   Databases      │
                       │                  │
                       │ - PostgreSQL     │
                       │ - Qdrant Vector  │
                       └──────────────────┘
```

## Features

### Textbook Content
- 16 chapters following the Physical AI curriculum
- Theory, labs, quizzes, and assessments
- Instructor resources and lab manuals

### AI Integration
- RAG chatbot with textbook-grounded responses
- Content personalization engine
- Automated chapter generation

### Multilingual Support
- English and Urdu translations
- Translation API for content conversion

### Simulation Integration
- ROS2 labs and exercises
- Gazebo and Isaac Sim environments
- Hardware-in-the-loop capabilities

## Installation

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+

### Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd physical-ai-textbook
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
docker-compose up --build
```

4. Access the services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Chatbot API: http://localhost:8001

## Services

### Frontend (Docusaurus)
- Hosts the textbook content
- Provides navigation and search
- Integrates with backend services

### Backend (FastAPI)
- User authentication and management
- Content management API
- Personalization engine
- Translation services

### Chatbot (FastAPI)
- RAG system for textbook Q&A
- Vector database integration
- OpenAI API integration

## Development

### Adding New Chapters
1. Create a new markdown file in `frontend/docs/moduleX/`
2. Add to the sidebar configuration in `frontend/sidebars.js`
3. Update the database schema if needed

### Running Locally
For development, you can run services individually:

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn src.main:app --reload

# Chatbot
cd chatbot
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm start
```

## Deployment

The system is designed for containerized deployment using Docker Compose. For production:

1. Update the `.env` file with production configuration
2. Set up SSL certificates
3. Configure domain names and reverse proxy
4. Run `docker-compose up -d`

## API Documentation

API documentation is available at:
- Backend: http://localhost:8000/api/docs
- Chatbot: http://localhost:8001/docs

## Technologies Used

- **Frontend**: Docusaurus, React
- **Backend**: FastAPI, Python
- **Database**: PostgreSQL, Qdrant (Vector DB)
- **AI/ML**: OpenAI API, Sentence Transformers
- **Simulation**: ROS2, Gazebo, Isaac Sim
- **Containerization**: Docker, Docker Compose

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.
 

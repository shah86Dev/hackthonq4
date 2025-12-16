# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Development Environment Setup

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker and Docker Compose
- ROS2 Humble Hawksbill
- Isaac Sim (optional for development, required for testing)
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python manage.py migrate

# Start backend server
python manage.py runserver
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start Docusaurus development server
npm start
```

### 4. Chatbot Setup
```bash
# Navigate to chatbot directory
cd chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize vector store
python init_vector_store.py

# Start chatbot service
python app.py
```

## Running the Application

### Development Mode
```bash
# Terminal 1: Start backend
cd backend
source venv/bin/activate
python manage.py runserver

# Terminal 2: Start frontend
cd frontend
npm start

# Terminal 3: Start chatbot
cd chatbot
source venv/bin/activate
python app.py
```

### Using Docker Compose (Recommended)
```bash
# From the project root
docker-compose up --build
```

## Key Configuration

### Environment Variables
Create `.env` files in each service directory:

**Backend (.env):**
```
DATABASE_URL=postgresql://user:password@localhost/dbname
QDRANT_URL=http://qdrant:6333
NEON_DB_URL=postgresql://user:password@localhost/neon_db
SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-key
```

**Frontend (.env):**
```
BACKEND_API_URL=http://localhost:8000
CHATBOT_API_URL=http://localhost:8001
AUTH_API_URL=http://localhost:8000
```

**Chatbot (.env):**
```
QDRANT_URL=http://qdrant:6333
OPENAI_API_KEY=your-openai-key
BACKEND_API_URL=http://backend:8000
TEXTBOOK_CONTENT_PATH=/app/content
```

## Content Management

### Adding New Chapters
1. Create a new markdown file in `frontend/docs/chapters/`
2. Add metadata to follow the textbook structure
3. Update the sidebar configuration in `frontend/sidebars.js`
4. Run content validation: `npm run validate-content`

### Adding Lab Exercises
1. Create lab configuration in `simulation/ros2_ws/src/`
2. Add simulation assets to `simulation/isaac_sim/assets/`
3. Update the lab manifest with instructions and validation criteria

## Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

### End-to-End Tests
```bash
# With all services running
npm run test:e2e
```

## Deployment

### Local Deployment
```bash
# Build frontend
cd frontend
npm run build

# Deploy to GitHub Pages
npm run deploy
```

### Production Deployment
1. Set up production environment variables
2. Build all services: `npm run build:all`
3. Deploy to your preferred hosting platform

## Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 3000 (frontend), 8000 (backend), 8001 (chatbot) are available
2. **Database connection**: Verify PostgreSQL and Neon connections
3. **Vector store**: Ensure Qdrant is running and accessible
4. **ROS2 environment**: Source ROS2 setup before running simulation components

### Development Tips
- Use `npm run dev` for hot reloading during frontend development
- Access the admin panel at `/admin` for content management
- Check the logs in each service's respective log directory
- Use the API documentation at `/api/docs` for backend endpoints
# API Contracts: Physical AI & Humanoid Robotics Textbook

## Content API

### GET /api/chapters
**Description**: Retrieve list of all textbook chapters
**Authentication**: Optional (required for personalized content)
**Parameters**:
- `module` (optional): Filter by module (ros2, gazebo, isaac, vla)
- `difficulty` (optional): Filter by difficulty level
- `personalize` (optional): Whether to return personalized content (default: false)

**Response**:
```json
{
  "chapters": [
    {
      "id": "string",
      "title": "string",
      "module": "string",
      "difficulty_level": "string",
      "learning_objectives": ["string"],
      "estimated_reading_time": "number"
    }
  ]
}
```

### GET /api/chapters/{chapterId}
**Description**: Retrieve specific chapter content
**Authentication**: Optional (required for personalized content)
**Parameters**:
- `language` (optional): Language preference (en, ur, default: en)
- `personalize` (optional): Whether to return personalized content (default: false)

**Response**:
```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "module": "string",
  "difficulty_level": "string",
  "learning_objectives": ["string"],
  "lab_exercises": ["LabExercise"],
  "quizzes": ["Quiz"],
  "instructor_resources": "InstructorResources",
  "next_chapter": "string"
}
```

### GET /api/chapters/{chapterId}/labs
**Description**: Retrieve lab exercises for a specific chapter
**Authentication**: Required for full access
**Parameters**:
- `language` (optional): Language preference (en, ur, default: en)

**Response**:
```json
{
  "lab_exercises": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "simulation_environment": "string",
      "instructions": "string",
      "expected_outcomes": ["string"],
      "estimated_duration": "number"
    }
  ]
}
```

## User API

### POST /api/auth/register
**Description**: Register a new user
**Authentication**: None
**Body**:
```json
{
  "email": "string",
  "password": "string",
  "name": "string",
  "role": "student|instructor",
  "background": "string"
}
```

**Response**:
```json
{
  "user_id": "string",
  "email": "string",
  "name": "string",
  "role": "string",
  "access_token": "string",
  "refresh_token": "string"
}
```

### POST /api/auth/login
**Description**: Login user
**Authentication**: None
**Body**:
```json
{
  "email": "string",
  "password": "string"
}
```

**Response**:
```json
{
  "user_id": "string",
  "email": "string",
  "role": "string",
  "access_token": "string",
  "refresh_token": "string"
}
```

### GET /api/user/profile
**Description**: Get user profile
**Authentication**: Required (JWT)
**Response**:
```json
{
  "id": "string",
  "email": "string",
  "name": "string",
  "role": "string",
  "background": "string",
  "preferred_language": "string",
  "personalization_enabled": "boolean",
  "created_at": "timestamp"
}
```

### PUT /api/user/profile
**Description**: Update user profile
**Authentication**: Required (JWT)
**Body**:
```json
{
  "name": "string",
  "background": "string",
  "preferred_language": "string",
  "personalization_enabled": "boolean"
}
```

## Learning Progress API

### GET /api/user/progress
**Description**: Get user's learning progress
**Authentication**: Required (JWT)
**Response**:
```json
{
  "progress": [
    {
      "chapter_id": "string",
      "chapter_title": "string",
      "status": "string",
      "progress_percentage": "number",
      "time_spent": "number",
      "completed_at": "timestamp"
    }
  ]
}
```

### POST /api/user/progress/{chapterId}
**Description**: Update user's progress for a chapter
**Authentication**: Required (JWT)
**Body**:
```json
{
  "current_position": "number",
  "status": "not_started|in_progress|completed"
}
```

**Response**:
```json
{
  "chapter_id": "string",
  "status": "string",
  "progress_percentage": "number",
  "last_updated": "timestamp"
}
```

## Quiz API

### GET /api/quizzes/{quizId}
**Description**: Retrieve quiz questions
**Authentication**: Required
**Parameters**:
- `language` (optional): Language preference (en, ur, default: en)
- `randomize` (optional): Whether to randomize questions (default: true)

**Response**:
```json
{
  "id": "string",
  "title": "string",
  "questions": [
    {
      "id": "string",
      "type": "string",
      "question_text": "string",
      "options": ["string"],
      "difficulty_level": "string"
    }
  ],
  "time_limit": "number"
}
```

### POST /api/quizzes/{quizId}/submit
**Description**: Submit quiz answers
**Authentication**: Required
**Body**:
```json
{
  "answers": [
    {
      "question_id": "string",
      "answer_text": "string"
    }
  ]
}
```

**Response**:
```json
{
  "quiz_id": "string",
  "score": "number",
  "total_questions": "number",
  "correct_answers": "number",
  "feedback": [
    {
      "question_id": "string",
      "is_correct": "boolean",
      "explanation": "string"
    }
  ]
}
```

## RAG Chatbot API

### POST /api/chat
**Description**: Send message to RAG chatbot
**Authentication**: Optional (for registered users)
**Body**:
```json
{
  "message": "string",
  "session_id": "string (optional)",
  "language": "en|ur (optional)"
}
```

**Response**:
```json
{
  "response": "string",
  "session_id": "string",
  "source_chunks": ["string"],
  "is_grounding_valid": "boolean"
}
```

### GET /api/chat/{sessionId}
**Description**: Get chat history for a session
**Authentication**: Required for registered user sessions
**Response**:
```json
{
  "session_id": "string",
  "messages": [
    {
      "id": "string",
      "sender": "user|assistant",
      "content": "string",
      "timestamp": "timestamp"
    }
  ]
}
```

## Translation API

### POST /api/translate
**Description**: Translate content
**Authentication**: Required
**Body**:
```json
{
  "text": "string",
  "source_lang": "string",
  "target_lang": "ur",
  "context": "string (optional, for domain-specific translation)"
}
```

**Response**:
```json
{
  "translated_text": "string",
  "confidence": "number"
}
```

## Instructor API

### GET /api/instructor/resources/{chapterId}
**Description**: Get instructor resources for a chapter
**Authentication**: Required (instructor role)
**Parameters**:
- `language` (optional): Language preference (en, ur, default: en)

**Response**:
```json
{
  "slides": ["string"],
  "assessment_bank": ["Question"],
  "teaching_notes": "string",
  "lab_setup_guides": ["string"]
}
```
# Data Model: Physical AI & Humanoid Robotics – Full University Textbook

## Entity: Chapter
- **id**: string (unique identifier)
- **title**: string (chapter title)
- **content**: string (markdown content)
- **content_urdu**: string (Urdu translation of content)
- **module**: string (ROS2, Gazebo, Isaac, VLA)
- **difficulty_level**: enum (beginner, intermediate, advanced)
- **learning_objectives**: array of string
- **prerequisites**: array of string (chapter IDs)
- **lab_exercises**: array of LabExercise
- **quizzes**: array of Quiz
- **instructor_resources**: InstructorResources
- **personalization_rules**: array of PersonalizationRule
- **metadata**: object (for AI indexing)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: LabExercise
- **id**: string (unique identifier)
- **title**: string (lab title)
- **description**: string (lab description)
- **simulation_environment**: enum (isaac_sim, gazebo, unity)
- **ros2_package**: string (ROS2 package name for the lab)
- **instructions**: string (step-by-step instructions)
- **instructions_urdu**: string (Urdu translation)
- **expected_outcomes**: array of string
- **difficulty_level**: enum (beginner, intermediate, advanced)
- **estimated_duration**: number (in minutes)
- **validation_criteria**: array of string
- **assets**: array of string (file paths to required assets)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: Quiz
- **id**: string (unique identifier)
- **title**: string (quiz title)
- **questions**: array of Question
- **passing_score**: number (percentage required to pass)
- **time_limit**: number (in minutes, 0 if no limit)
- **randomize_questions**: boolean
- **feedback_mode**: enum (immediate, delayed, none)
- **difficulty_level**: enum (beginner, intermediate, advanced)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: Question
- **id**: string (unique identifier)
- **type**: enum (multiple_choice, true_false, short_answer, essay)
- **question_text**: string
- **question_urdu**: string (Urdu translation)
- **options**: array of string (for multiple choice)
- **correct_answer**: string | array of string
- **explanation**: string (explanation of correct answer)
- **explanation_urdu**: string (Urdu translation)
- **difficulty_level**: enum (beginner, intermediate, advanced)
- **tags**: array of string (topic tags)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: User
- **id**: string (unique identifier)
- **email**: string (email address)
- **name**: string (full name)
- **role**: enum (student, instructor, admin)
- **background**: string (educational/professional background)
- **preferred_language**: enum (english, urdu)
- **personalization_enabled**: boolean
- **learning_history**: array of LearningEvent
- **chapter_progress**: array of ChapterProgress
- **quiz_results**: array of QuizResult
- **preferences**: object (user preferences)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: ChapterProgress
- **user_id**: string (reference to User)
- **chapter_id**: string (reference to Chapter)
- **status**: enum (not_started, in_progress, completed)
- **current_position**: number (current position in chapter)
- **time_spent**: number (in seconds)
- **last_accessed**: timestamp
- **personalization_level**: number (0-100 scale)
- **completed_at**: timestamp (nullable)

## Entity: QuizResult
- **user_id**: string (reference to User)
- **quiz_id**: string (reference to Quiz)
- **score**: number (percentage score)
- **attempts**: number (number of attempts)
- **answers**: array of UserAnswer
- **completed_at**: timestamp
- **time_taken**: number (in seconds)

## Entity: UserAnswer
- **question_id**: string (reference to Question)
- **answer_text**: string (user's answer)
- **is_correct**: boolean
- **points_earned**: number
- **feedback_received**: string

## Entity: LearningEvent
- **user_id**: string (reference to User)
- **event_type**: enum (chapter_view, lab_start, lab_complete, quiz_start, quiz_complete)
- **entity_id**: string (ID of the entity involved)
- **entity_type**: enum (chapter, lab, quiz)
- **timestamp**: timestamp
- **metadata**: object (additional event data)

## Entity: InstructorResources
- **id**: string (unique identifier)
- **slides**: array of string (file paths to slide decks)
- **slides_urdu**: array of string (Urdu slide translations)
- **assessment_bank**: array of Question
- **teaching_notes**: string (instructor notes)
- **teaching_notes_urdu**: string (Urdu translation)
- **lab_setup_guides**: array of string (file paths)
- **lab_setup_guides_urdu**: string (Urdu translation)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: PersonalizationRule
- **id**: string (unique identifier)
- **trigger_condition**: string (condition that triggers personalization)
- **rule_type**: enum (difficulty_adjustment, example_substitution, content_reordering)
- **target_content**: string (ID of content to modify)
- **adjustment_parameters**: object (parameters for personalization)
- **user_background_match**: string (background that triggers this rule)
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: ChatSession
- **id**: string (unique identifier)
- **user_id**: string (nullable - for anonymous users)
- **session_token**: string (for anonymous sessions)
- **messages**: array of ChatMessage
- **created_at**: timestamp
- **updated_at**: timestamp

## Entity: ChatMessage
- **id**: string (unique identifier)
- **session_id**: string (reference to ChatSession)
- **sender**: enum (user, assistant)
- **content**: string (message content)
- **timestamp**: timestamp
- **source_chunks**: array of string (IDs of content chunks used for response)
- **is_grounding_valid**: boolean (whether response is properly grounded in textbook)

## Entity: ContentChunk
- **id**: string (unique identifier)
- **chapter_id**: string (reference to Chapter)
- **content**: string (text content for RAG)
- **content_urdu**: string (Urdu translation)
- **embedding**: array of number (vector embedding)
- **chunk_type**: enum (theory, lab, quiz, reference)
- **metadata**: object (for AI indexing)
- **created_at**: timestamp
- **updated_at**: timestamp
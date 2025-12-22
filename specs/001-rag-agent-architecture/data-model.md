# Data Model: RAG-Enabled Agent Architecture

## Overview
This document defines the data structures and relationships for the RAG-enabled agent architecture.

## Core Entities

### UserQuery
**Description**: Represents a user's input to the agent system

**Fields**:
- `id` (string): Unique identifier for the query
- `content` (string): The actual query text from the user
- `timestamp` (datetime): When the query was received
- `session_id` (string): Identifier for the conversation session
- `metadata` (object): Additional context information

**Validation**:
- `content` must not be empty
- `id` must be unique within the system

### Intent
**Description**: The interpreted purpose or goal extracted from a user query by Claude planner

**Fields**:
- `id` (string): Unique identifier for the intent
- `query_id` (string): Reference to the original UserQuery
- `type` (string): The category of intent (e.g., "rag_search", "script_execution")
- `parameters` (object): Specific parameters for the intent
- `confidence` (float): Confidence score for the intent classification

**Validation**:
- `type` must be a valid skill identifier
- `confidence` must be between 0 and 1

### Skill
**Description**: A discrete capability that can be invoked to handle specific intents

**Fields**:
- `name` (string): Unique identifier for the skill
- `description` (string): Human-readable description of what the skill does
- `parameters` (object): Expected input parameters for the skill
- `execution_method` (string): How the skill executes ("local_script", "mcp")
- `script_path` (string): Path to the script for local execution (if applicable)

**Validation**:
- `name` must be unique across all skills
- `execution_method` must be one of the allowed values

### ExecutionResult
**Description**: The output from a skill execution

**Fields**:
- `id` (string): Unique identifier for the result
- `skill_name` (string): Name of the skill that produced the result
- `query_id` (string): Reference to the original query
- `status` (string): Execution status ("success", "error", "timeout")
- `output` (object): The actual result data (serialized)
- `execution_time` (float): Time taken to execute in seconds
- `metadata` (object): Additional execution information

**Validation**:
- `status` must be one of the allowed values
- `output` must be JSON-serializable

### VectorStoreEntry
**Description**: An entry in the Qdrant vector store for RAG functionality

**Fields**:
- `id` (string): Unique identifier for the vector entry
- `content` (string): The text content of the entry
- `embedding` (array): The vector embedding of the content
- `metadata` (object): Additional metadata about the content
- `source` (string): Source identifier for the content

**Validation**:
- `embedding` must be a valid vector array
- `content` must not be empty

### APISession
**Description**: Represents a user session interacting with the API

**Fields**:
- `session_id` (string): Unique identifier for the session
- `user_id` (string): Identifier for the user (if available)
- `created_at` (datetime): When the session was created
- `last_activity` (datetime): When the session was last used
- `context_history` (array): Previous conversation context

**Validation**:
- `session_id` must be unique
- `context_history` should be limited in size

## Relationships

1. **UserQuery** → **Intent**: One-to-one (each query results in one primary intent)
2. **Intent** → **Skill**: One-to-one (each intent maps to one skill)
3. **Skill** → **ExecutionResult**: One-to-many (each skill can produce multiple results)
4. **UserQuery** → **ExecutionResult**: One-to-many (one query may trigger multiple skill executions)
5. **APISession** → **UserQuery**: One-to-many (one session can have multiple queries)

## State Transitions

### UserQuery States
- `received` → `intent_identified` → `skill_executed` → `response_delivered`

### ExecutionResult States
- `pending` → `executing` → (`success` | `error` | `timeout`)

## Validation Rules

1. All timestamps must be in ISO 8601 format
2. All IDs must follow UUID format or similar globally unique identifier
3. Content fields must be properly sanitized to prevent injection attacks
4. Context history must be limited to prevent excessive memory usage
5. Execution times must be monitored and logged for performance tracking

## Serialization Format

All data entities will be serialized as JSON objects with consistent field naming using snake_case convention.
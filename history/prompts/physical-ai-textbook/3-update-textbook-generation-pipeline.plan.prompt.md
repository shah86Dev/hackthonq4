---
id: 3
title: update-textbook-generation-pipeline
stage: plan
date: 2025-12-16
surface: agent
model: claude-opus-4-5-20251101
feature: physical-ai-textbook
branch: main
user: user
command: /sp.plan
labels: ["textbook-generation","pipeline","architecture"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-physical-ai-textbook/plan.md
tests:
 - null
---

## Prompt

/sp.plan

Generation Pipeline:

Step 1: Load JSON Book Spec
- Validate schema
- Resolve modules, chapters, labs

Step 2: Expand Modules
- Convert each module → chapters
- Map weeks → chapters

Step 3: Expand Chapters
For each chapter:
- Generate theory sections
- Generate diagrams (textual description)
- Generate ROS 2 / Isaac examples
- Generate labs
- Generate assessments

Step 4: AI-Native Enhancements
- Attach RAG metadata per section
- Mark personalization & translation flags
- Add sim-to-real annotations

Step 5: Output Assembly
- /docs/module-x/chapter-y.md
- /docs/labs/
- /docs/capstone/
- /docs/assessments/
- /rag/embeddings.json

Step 6: Validation
- Learning outcomes coverage
- Hardware feasibility
- Weekly alignment check

## Response snapshot

Updated the existing plan.md file to include the detailed AI textbook generation pipeline with 6 steps: Load JSON Book Spec, Expand Modules, Expand Chapters, AI-Native Enhancements, Output Assembly, and Validation. The plan now includes specific details about content generation, RAG integration, and output structure.

## Outcome

- ✅ Impact: Enhanced plan with detailed generation pipeline for AI textbook
- 🧪 Tests: null
- 📁 Files: Updated plan.md with comprehensive pipeline steps
- 🔁 Next prompts: Implementation of the generation pipeline
- 🧠 Reflection: Pipeline covers the complete flow from JSON input to Docusaurus output with AI enhancements

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Begin implementation of Step 1 of the pipeline
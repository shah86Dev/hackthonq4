---
id: 1
title: Physical AI Textbook Constitution
stage: constitution
date: 2025-12-16
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: main
user: Claude
command: /sp.constitution
labels: [constitution, textbook, physical-ai, humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests: []
---

## Prompt

```
/sp.contitution /sp.constitution
Project: "Physical AI & Humanoid Robotics – Panaversity Textbook"

Principles:
1. Everything is AI-Native: Every chapter must be fully compatible with RAG, embeddings, and AI-assisted learning.
2. Coursework Must Match Panaversity's Physical AI Curriculum:
   - ROS 2 (Module 1)
   - Gazebo & Unity (Module 2)
   - NVIDIA Isaac (Module 3)
   - VLA Models (Module 4)
3. Simulation-First Learning: All labs must run on Isaac Sim/Gazebo.
4. Real Hardware Integration: Must include Jetson kits, RealSense, Unitree proxies.
5. Cloud + Local Dual Architecture: Content for RTX Workstations and AWS g5/g6 instances.
6. Students Can Personalize Learning: Each chapter supports personalization and Urdu translation.
7. Spec-Driven: All chapters generated from a JSON spec template.
8. Agent Ready: Every chapter is optimised for the RAG Chatbot.
9. Subagents & Skills: Book includes reusable intelligence with Claude Code subagents.
10. Modular: Each chapter has Theory, Labs, Assessments, Projects, Glossary, References.
```

## Response snapshot

Updated the constitution file (.specify/memory/constitution.md) with all 10 project principles for the Physical AI & Humanoid Robotics textbook. The updated constitution now includes:

1. Everything is AI-Native: Every chapter must be fully compatible with RAG, embeddings, and AI-assisted learning
2. Coursework Must Match Panaversity's Physical AI Curriculum: Content aligned with ROS 2, Gazebo & Unity, NVIDIA Isaac, and VLA Models modules
3. Simulation-First Learning: All labs must run on Isaac Sim/Gazebo simulation environments
4. Real Hardware Integration: Content includes practical integration with Jetson kits, RealSense sensors, and Unitree robot proxies
5. Cloud + Local Dual Architecture: Content accessible on both RTX Workstations and AWS g5/g6 instances
6. Students Can Personalize Learning: Chapters support personalization and Urdu translation capabilities
7. Spec-Driven: All chapters generated from JSON spec template following structured approach
8. Agent Ready: Every chapter optimized for RAG Chatbot retrieval and generation
9. Subagents & Skills: Book includes reusable intelligence with Claude Code subagents
10. Modular Structure: Each chapter has Theory, Labs, Assessments, Projects, Glossary, and References

The constitution also includes additional requirements, development workflow, and governance sections.

## Outcome

- ✅ Impact: Updated project constitution with 10 specific principles for Physical AI textbook
- 🧪 Tests: No tests required for constitution update
- 📁 Files: Updated .specify/memory/constitution.md with all project principles
- 🔁 Next prompts: Ready to proceed with feature specification based on new constitution
- 🧠 Reflection: Constitution now properly reflects the specific requirements for the Physical AI & Humanoid Robotics textbook project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin feature specification following new constitution
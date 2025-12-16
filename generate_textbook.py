#!/usr/bin/env python3
"""
AI Textbook Generation Pipeline
Processes panavercity_physical_ai_book.json and generates:
- Docusaurus-compatible documentation
- Lab manuals
- Capstone specifications
- RAG-ready embeddings
- Metadata for learning outcomes
"""

import json
import os
from pathlib import Path
import re
from typing import Dict, List, Any


def load_book_json(file_path: str) -> Dict:
    """Load the textbook JSON specification."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_filename(name: str) -> str:
    """Convert a title to a safe filename."""
    # Remove special characters and replace spaces with hyphens
    name = re.sub(r'[^\w\s-]', '', name.lower())
    name = re.sub(r'[-\s]+', '-', name.strip())
    return name


def create_directory_structure():
    """Create the required directory structure."""
    dirs = [
        'frontend/docs',
        'frontend/docs/labs',
        'frontend/docs/capstone',
        'frontend/docs/assessments',
        'frontend/rag',
        'frontend/meta'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def generate_module_directories(book_data: Dict):
    """Generate module directories."""
    for module in book_data['modules']:
        module_dir = f"frontend/docs/module{module['id'][-1]}"
        Path(module_dir).mkdir(exist_ok=True)
        print(f"Created module directory: {module_dir}")


def generate_chapter_markdown(book_data: Dict):
    """Generate chapter markdown files for Docusaurus."""
    for module in book_data['modules']:
        module_id = module['id']
        module_num = module_id[-1]  # Extract number from module1, module2, etc.

        for chapter in module['chapters']:
            chapter_id = chapter['id']
            chapter_title = chapter['title']
            chapter_subtitle = chapter.get('subtitle', '')

            # Create filename
            safe_title = sanitize_filename(chapter_title)
            filename = f"{module_num}-{safe_title}.md"
            filepath = f"frontend/docs/module{module_num}/{filename}"

            # Generate markdown content
            markdown_content = f"""---
id: {chapter_id}
title: {chapter_title}
sidebar_label: {chapter_title}
---

# {chapter_title}

{chapter_subtitle}

## Learning Outcomes

{'\n\n'.join([f'- {outcome}' for outcome in chapter['learning_outcomes']])}

## Prerequisites

{'\n\n'.join([f'- {prereq}' for prereq in chapter.get('prerequisites', [])])}

"""

            # Add sections
            for section in chapter['sections']:
                markdown_content += f"""
## {section['title']}

{section['content']}

"""

                # Add diagrams
                for diagram in section.get('diagrams', []):
                    markdown_content += f"""
### {diagram['title']}

{diagram['description']}

"""

                # Add code examples
                for example in section.get('examples', []):
                    markdown_content += f"""
### {example['title']}

```{example.get('language', 'text')}
{example['code']}
```

"""

            # Write the markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Generated chapter: {filepath}")


def generate_labs(book_data: Dict):
    """Generate lab manuals from the book data."""
    lab_content = "# Lab Manuals\n\n"

    for module in book_data['modules']:
        module_num = module['id'][-1]
        lab_content += f"## Module {module_num}: {module['title']}\n\n"

        for chapter in module['chapters']:
            chapter_title = chapter['title']
            lab_content += f"### {chapter_title}\n\n"

            for section in chapter['sections']:
                for lab in section.get('labs', []):
                    lab_content += f"""
#### {lab['title']}

**Duration**: {lab['duration']} minutes
**Difficulty**: {lab['difficulty']}

{lab['description']}

**Instructions:**
{'\n'.join([f"- {instruction}" for instruction in lab['instructions']])}

**Expected Outcomes:**
{'\n'.join([f"- {outcome}" for outcome in lab['expected_outcomes']])}

---
"""

    # Write the main labs file
    with open("frontend/docs/labs/lab-manual.md", 'w', encoding='utf-8') as f:
        f.write(lab_content)

    print("Generated lab manual: frontend/docs/labs/lab-manual.md")


def generate_capstone(book_data: Dict):
    """Generate capstone project specification."""
    capstone = book_data['capstone']

    capstone_content = f"""---
id: capstone-project
title: {capstone['title']}
sidebar_label: {capstone['title']}
---

# {capstone['title']}

{capstone['description']}

## Duration
{capstone['duration']}

## Objectives

{'\n'.join([f'- {obj}' for obj in capstone['objectives']])}

## Requirements

{'\n'.join([f'- {req}' for req in capstone['requirements']])}

## Evaluation Criteria

{'\n'.join([f'- {crit}' for crit in capstone['evaluation_criteria']])}

"""

    with open("frontend/docs/capstone/capstone-project.md", 'w', encoding='utf-8') as f:
        f.write(capstone_content)

    print("Generated capstone: frontend/docs/capstone/capstone-project.md")


def generate_assessments(book_data: Dict):
    """Generate assessment materials."""
    assessment_content = "# Assessments\n\n"

    for module in book_data['modules']:
        module_num = module['id'][-1]
        assessment_content += f"## Module {module_num}: {module['title']}\n\n"

        for chapter in module['chapters']:
            chapter_title = chapter['title']
            assessment_content += f"### {chapter_title}\n\n"

            for section in chapter['sections']:
                for assessment in section.get('assessments', []):
                    assessment_content += f"""
#### {assessment['question']}

**Type**: {assessment['type']}

"""
                    if 'options' in assessment:
                        for i, option in enumerate(assessment['options'], 1):
                            assessment_content += f" {i}. {option}\n"

                    if 'explanation' in assessment:
                        assessment_content += f"\n**Explanation**: {assessment['explanation']}\n"
                    elif 'rubric' in assessment:
                        assessment_content += f"\n**Rubric**: {assessment['rubric']}\n"

                    assessment_content += "\n---\n"

    with open("frontend/docs/assessments/assessments.md", 'w', encoding='utf-8') as f:
        f.write(assessment_content)

    print("Generated assessments: frontend/docs/assessments/assessments.md")


def generate_rag_embeddings(book_data: Dict):
    """Generate RAG-ready embeddings input."""
    embeddings = []

    # Process each section as a content chunk for RAG
    for module in book_data['modules']:
        module_title = module['title']

        for chapter in module['chapters']:
            chapter_title = chapter['title']

            for section in chapter['sections']:
                section_title = section['title']
                content = section['content']

                # Create embedding entry
                embedding_entry = {
                    'id': f"{section['id']}",
                    'module': module_title,
                    'chapter': chapter_title,
                    'section': section_title,
                    'content': content,
                    'metadata': {
                        'module_id': module['id'],
                        'chapter_id': chapter['id'],
                        'section_id': section['id'],
                        'learning_outcomes': chapter['learning_outcomes'],
                        'prerequisites': chapter.get('prerequisites', [])
                    }
                }

                embeddings.append(embedding_entry)

    # Write embeddings to JSON file
    with open("frontend/rag/embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)

    print("Generated RAG embeddings: frontend/rag/embeddings.json")


def generate_metadata(book_data: Dict):
    """Generate learning outcomes and specifications metadata."""
    metadata = {
        'book_title': book_data['title'],
        'book_subtitle': book_data['subtitle'],
        'version': book_data['version'],
        'description': book_data['description'],
        'total_modules': book_data['metadata']['total_modules'],
        'total_chapters': book_data['metadata']['total_chapters'],
        'total_weeks': book_data['metadata']['total_weeks'],
        'target_audience': book_data['metadata']['target_audience'],
        'prerequisites': book_data['metadata']['prerequisites'],
        'learning_outcomes_by_module': {},
        'ai_native_features': book_data['metadata']['ai_native_features']
    }

    # Extract learning outcomes by module
    for module in book_data['modules']:
        module_id = module['id']
        module_outcomes = []

        for chapter in module['chapters']:
            for outcome in chapter['learning_outcomes']:
                module_outcomes.append({
                    'chapter_id': chapter['id'],
                    'chapter_title': chapter['title'],
                    'outcome': outcome
                })

        metadata['learning_outcomes_by_module'][module_id] = module_outcomes

    # Write metadata to JSON file
    with open("frontend/meta/learning-outcomes-specs.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Generated metadata: frontend/meta/learning-outcomes-specs.json")


def generate_main_textbook_index(book_data: Dict):
    """Generate main index and navigation files."""
    # Generate main index
    index_content = f"""---
id: intro
title: {book_data['title']}
sidebar_label: Introduction
slug: /
---

# {book_data['title']}

{book_data['subtitle']}

**Version**: {book_data['version']}

{book_data['description']}

## Course Structure

This textbook is organized into {book_data['metadata']['total_modules']} modules spanning {book_data['metadata']['total_weeks']} weeks:

"""

    for i, module in enumerate(book_data['modules'], 1):
        index_content += f"### Module {i}: {module['title']} ({module['weeks']} weeks)\n\n"
        for chapter in module['chapters']:
            index_content += f"- [{chapter['title']}]({{ '/docs/module{i}/{sanitize_filename(chapter['title'])}' | relative_url }})\n"
        index_content += "\n"

    index_content += f"""
## Capstone Project

[{book_data['capstone']['title']}]({{ '/docs/capstone/capstone-project' | relative_url }})

## Additional Resources

- [Lab Manuals]({{ '/docs/labs/lab-manual' | relative_url }})
- [Assessments]({{ '/docs/assessments/assessments' | relative_url }})

"""

    with open("frontend/docs/intro.md", 'w', encoding='utf-8') as f:
        f.write(index_content)

    print("Generated textbook index: frontend/docs/intro.md")


def main():
    """Main execution function."""
    print("Starting AI Textbook Generation Pipeline...")

    # Load the book data
    book_data = load_book_json('panavercity_physical_ai_book.json')
    print(f"Loaded book: {book_data['title']}")

    # Create directory structure
    create_directory_structure()

    # Generate module directories
    generate_module_directories(book_data)

    # Generate chapter markdown files
    generate_chapter_markdown(book_data)

    # Generate lab manuals
    generate_labs(book_data)

    # Generate capstone project
    generate_capstone(book_data)

    # Generate assessments
    generate_assessments(book_data)

    # Generate RAG embeddings
    generate_rag_embeddings(book_data)

    # Generate metadata
    generate_metadata(book_data)

    # Generate main index
    generate_main_textbook_index(book_data)

    print("\nAI Textbook Generation Pipeline COMPLETE!")
    print("\nGenerated outputs:")
    print("- /frontend/docs (textbook chapters)")
    print("- /frontend/docs/labs (hands-on labs)")
    print("- /frontend/docs/capstone (capstone project)")
    print("- /frontend/rag (Qdrant embeddings input)")
    print("- /frontend/meta (learning outcomes & specs)")


if __name__ == "__main__":
    main()
# Code Explainer Skill

A reusable skill that automatically explains code with diagrams and analogies when triggered by "how does this work?" questions.

## Trigger Condition
This skill is automatically activated when the user asks: "how does this work?"

## Purpose
To provide clear, visual explanations of code functionality using analogies, diagrams, and step-by-step breakdowns that make complex code concepts accessible and understandable.

## Step-by-Step Instructions

### 1. Analyze the Code
- Identify the main components and structure of the code
- Determine the programming language and framework being used
- Identify key functions, classes, methods, and their relationships

### 2. Break Down the Logic
- Explain the overall purpose of the code in simple terms
- Identify the input, processing, and output components
- Trace the flow of execution step-by-step
- Highlight any loops, conditions, or complex logic

### 3. Create Visual Diagrams
- Use ASCII art or mermaid diagrams to illustrate:
  - Data flow through the system
  - Class hierarchies or object relationships
  - Function call sequences
  - State transitions
  - Component interactions

### 4. Provide Real-World Analogies
- Compare code concepts to everyday objects or processes
- Use analogies that match the user's likely background or interests
- Relate abstract programming concepts to concrete examples
- Make connections between code structure and familiar systems

### 5. Explain with Examples
- Provide concrete examples of how the code would behave with specific inputs
- Show expected outputs or results
- Include edge cases and error handling if relevant
- Demonstrate with sample data or scenarios

### 6. Highlight Key Points
- Point out important implementation details
- Explain why certain approaches were chosen
- Mention any performance or security considerations
- Note any potential issues or improvements

## Example Usage Pattern

When the user asks "how does this work?", respond with:

```
## Code Explanation: [Component Name]

### Overview
[Simple explanation of what the code does]

### Visual Representation
[ASCII diagram or mermaid flowchart showing the structure/flow]

### Real-World Analogy
[Relatable analogy comparing the code to a familiar concept]

### Step-by-Step Breakdown
1. [Step 1] - [Explanation]
2. [Step 2] - [Explanation]
3. [Step 3] - [Explanation]

### Example
[Concrete example with sample input/output]

### Key Points
- [Important point 1]
- [Important point 2]
- [Important point 3]
```

## Sample Response Template

```
## Code Explanation: [Title]

### Overview
[One sentence summary of what the code does]

### How It Works
[Detailed explanation broken into logical sections]

### Visual Representation
```mermaid
[Flowchart, sequence diagram, or other visual representation]
```

### Real-World Analogy
Think of this code like [analogous real-world system/process]. Just as [real-world example], the code [how it relates to the code].

### Step-by-Step Flow
1. [First step] - [What happens and why]
2. [Second step] - [What happens and why]
3. [Third step] - [What happens and why]

### Example in Action
Input: [example input]
Process: [what happens step-by-step]
Output: [expected result]

### Key Insights
- [Important insight about the implementation]
- [Performance or design consideration]
- [Potential use cases or applications]
```

## Activation
This skill automatically activates when the user asks "how does this work?" about any code snippet, function, class, or system. Always provide visual elements (diagrams, flowcharts) and relatable analogies to enhance understanding.
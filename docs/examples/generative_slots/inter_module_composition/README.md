# Inter-Module Composition Examples

This subdirectory demonstrates how to compose multiple generative functions together to build complex workflows.

## Files

### summarizers.py
Defines reusable summarization functions that can be composed with other modules.

### decision_aides.py
Implements decision-making functions that can work with summarized content.

### summarize_and_decide.py
Shows how to compose summarizers and decision aides into a complete workflow.

## Concepts Demonstrated

- **Module Composition**: Building complex workflows from simple functions
- **Reusability**: Creating reusable generative components
- **Pipeline Design**: Chaining multiple LLM operations
- **Separation of Concerns**: Keeping different functionalities in separate modules

## Pattern

```python
# Module 1: summarizers.py
@generative
def summarize_text(text: str) -> str:
    """Summarize the given text."""

# Module 2: decision_aides.py
@generative
def make_decision(summary: str) -> str:
    """Make a decision based on the summary."""

# Composition: summarize_and_decide.py
with start_session() as m:
    summary = summarize_text(m, text=long_document)
    decision = make_decision(m, summary=summary)
```

## Benefits

- **Modularity**: Each function has a single responsibility
- **Testability**: Individual components can be tested separately
- **Maintainability**: Changes to one module don't affect others
- **Composability**: Mix and match functions for different workflows
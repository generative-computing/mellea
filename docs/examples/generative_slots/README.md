# Generative Slots Examples

This directory contains examples demonstrating the `@generative` decorator for creating type-safe, composable LLM functions.

## Files

### generative_slots.py
Basic introduction to generative slots with sentiment classification and text summarization.

**Key Features:**
- Using `@generative` decorator for type-safe LLM functions
- Literal types for constrained outputs
- Docstrings as prompts for the LLM
- Simple function composition

### generate_with_context.py
Shows how to use generative slots with custom context and grounding information.

### generative_gsm8k.py
Demonstrates using generative slots for mathematical reasoning tasks (GSM8K dataset).

### generative_slots_with_requirements.py
Combines generative slots with requirements for validated outputs.

### investment_advice.py
A more complex example using generative slots for financial analysis.

### inter_module_composition/
Subdirectory with examples of composing multiple generative functions together.

## Concepts Demonstrated

- **Type Safety**: Using Python type hints for structured outputs
- **Docstring Prompts**: Leveraging docstrings as instructions to the LLM
- **Composition**: Building complex workflows from simple generative functions
- **Requirements**: Adding validation to generative functions
- **Context Management**: Using grounding context with generative slots

## Basic Usage

```python
from mellea import generative, start_session
from typing import Literal

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the given text."""

@generative
def generate_summary(text: str) -> str:
    """Generate a concise summary under 20 words."""

with start_session() as m:
    sentiment = classify_sentiment(m, text="I love this!")
    summary = generate_summary(m, text="Long document...")
```

## Related Documentation

- See `mellea/stdlib/components/genslot.py` for implementation
- See `docs/dev/mellea_library.md` for design philosophy
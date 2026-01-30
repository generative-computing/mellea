# Safety Examples

This directory contains examples of using Granite Guardian models for content safety and validation.

## Files

### guardian.py
Comprehensive examples of using the enhanced GuardianCheck requirement with Granite Guardian 3.3 8B.

**Key Features:**
- Multiple risk types (harm, jailbreak, social bias, etc.)
- Thinking mode for detailed reasoning
- Custom criteria for domain-specific safety
- Groundedness detection
- Function call hallucination detection
- Multiple backend support (Ollama, HuggingFace)

### guardian_huggingface.py
Using Guardian models with HuggingFace backend.

### repair_with_guardian.py
Combining Guardian safety checks with automatic repair.

## Concepts Demonstrated

- **Content Safety**: Detecting harmful, biased, or inappropriate content
- **Jailbreak Detection**: Identifying attempts to bypass safety measures
- **Groundedness**: Ensuring responses are factually grounded
- **Function Call Validation**: Detecting hallucinated tool calls
- **Multi-Risk Assessment**: Checking multiple safety criteria
- **Thinking Mode**: Getting detailed reasoning for safety decisions

## Available Risk Types

```python
from mellea.stdlib.requirements.safety.guardian import GuardianRisk

# Built-in risk types
GuardianRisk.HARM              # Harmful content
GuardianRisk.JAILBREAK         # Jailbreak attempts
GuardianRisk.SOCIAL_BIAS       # Social bias
GuardianRisk.GROUNDEDNESS      # Factual grounding
GuardianRisk.FUNCTION_CALL     # Function call hallucination
# ... and more
```

## Basic Usage

```python
from mellea import start_session
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk

# Create guardian with specific risk type
guardian = GuardianCheck(GuardianRisk.HARM, thinking=True)

# Use in validation
m = start_session()
m.chat("Write a professional email.")
is_safe = m.validate([guardian])

print(f"Content is safe: {is_safe[0]._result}")
if is_safe[0]._reason:
    print(f"Reasoning: {is_safe[0]._reason}")
```

## Advanced Usage

### Custom Criteria
```python
custom_guardian = GuardianCheck(
    custom_criteria="Check for inappropriate content in educational context"
)
```

### Groundedness Detection
```python
groundedness_guardian = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    thinking=True,
    context_text="Reference text for grounding check..."
)
```

### Function Call Validation
```python
function_guardian = GuardianCheck(
    GuardianRisk.FUNCTION_CALL,
    thinking=True,
    tools=[tool_definition]
)
```

### Multiple Guardians
```python
guardians = [
    GuardianCheck(GuardianRisk.HARM),
    GuardianCheck(GuardianRisk.JAILBREAK),
    GuardianCheck(GuardianRisk.SOCIAL_BIAS),
]
results = m.validate(guardians)
```

## Thinking Mode

Enable `thinking=True` to get detailed reasoning:
```python
guardian = GuardianCheck(GuardianRisk.HARM, thinking=True)
result = m.validate([guardian])
print(result[0]._reason)  # Detailed explanation
```

## Backend Support

- **Ollama**: `backend_type="ollama"` (default)
- **HuggingFace**: `backend_type="huggingface"`
- **Custom**: Pass your own backend instance

## Models

- Granite Guardian 3.0 2B
- Granite Guardian 3.3 8B (recommended)

## Related Documentation

- See `mellea/stdlib/requirements/safety/guardian.py` for implementation
- See `test/stdlib/requirements/` for more examples
- See IBM Granite Guardian documentation for model details
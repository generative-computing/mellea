# Instruct-Validate-Repair Examples

This directory demonstrates Mellea's core instruct-validate-repair paradigm for reliable LLM outputs.

## Files

### 101_email.py
The simplest example - using `m.instruct()` to generate an email.

**Key Features:**
- Basic session creation with `start_session()`
- Simple instruction without requirements
- Accessing the last prompt with `m.last_prompt()`

### 101_email_with_requirements.py
Adds requirements to constrain the output.

**Key Features:**
- Using string-based requirements
- Automatic validation and repair
- Ensuring output meets specified criteria

### 101_email_with_validate.py
Explicitly demonstrates the validation step.

**Key Features:**
- Separating generation and validation
- Using `m.validate()` to check requirements
- Understanding validation results

### 101_email_comparison.py
Compares outputs with and without requirements.

### advanced_email_with_validate_function.py
Shows how to use custom validation functions for complex requirements.

**Key Features:**
- Creating custom validation functions
- Using `simple_validate()` helper
- Combining multiple validation strategies

## Concepts Demonstrated

- **Instruct**: Generating outputs with natural language instructions
- **Validate**: Checking outputs against requirements
- **Repair**: Automatically fixing outputs that fail validation
- **Requirements**: Constraining outputs with natural language or functions
- **Sampling Strategies**: Using rejection sampling for reliable outputs

## Basic Pattern

```python
from mellea import start_session

# 1. Instruct
m = start_session()
result = m.instruct(
    "Write an email to invite interns to the office party.",
    requirements=[
        "Keep it under 50 words",
        "Include a date and time",
        "Be professional"
    ]
)

# 2. Validate (automatic with requirements)
# 3. Repair (automatic if validation fails)
print(result)
```

## Advanced Pattern

```python
from mellea.stdlib.requirements import simple_validate, req
from mellea.stdlib.sampling import RejectionSamplingStrategy

def check_length(text: str) -> bool:
    return len(text.split()) < 50

result = m.instruct(
    "Write an email...",
    requirements=[
        req("Under 50 words", validation_fn=simple_validate(check_length))
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

## Related Documentation

- See `mellea/stdlib/requirements/` for requirement types
- See `mellea/stdlib/sampling/` for sampling strategies
- See `docs/dev/mellea_library.md` for design philosophy
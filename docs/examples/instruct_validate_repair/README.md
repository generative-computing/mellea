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

### qiskit_code_validation/qiskit_code_validation.py
Advanced example demonstrating IVR pattern for Qiskit code generation with external validation.

**Key Features:**
- Integrating external validation tools (flake8-qiskit-migration)
- Automatic repair of deprecated Qiskit APIs
- Pre-condition validation of input code
- Custom validation functions for linters

**See:** [qiskit_code_validation/README.md](qiskit_code_validation/README.md) for full documentation and example prompts.

### multiturn_strategy_example.py
Demonstrates MultiTurnStrategy for conversational repair with validation feedback.

**Key Features:**
- Using ChatContext for multi-turn conversations
- Validation functions with detailed failure reasons
- Iterative improvement through conversational feedback
- Understanding when to use different repair strategies

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

## Sampling Strategies

Mellea provides three main sampling strategies for handling validation failures:

### RejectionSamplingStrategy
- **Use case**: Simple retry with the same prompt
- **Behavior**: Repeats the exact same instruction if validation fails
- **Best for**: Non-deterministic failures, simple requirements
- **Context**: Doesn't modify context between attempts

**Example:**
```python
from mellea.stdlib.sampling import RejectionSamplingStrategy

result = m.instruct(
    "Write an email...",
    requirements=["be formal", "under 50 words"],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

### RepairTemplateStrategy
- **Use case**: Single-turn repair with feedback
- **Behavior**: Adds validation failure reasons to the instruction and retries
- **Best for**: Simple tasks where feedback can be added to the instruction
- **Context**: Doesn't modify context, only the instruction

**Example:**
```python
from mellea.stdlib.sampling import RepairTemplateStrategy

result = m.instruct(
    "Write an email...",
    requirements=["be formal", word_count_req],
    strategy=RepairTemplateStrategy(loop_budget=3)
)
```

### MultiTurnStrategy
- **Use case**: Multi-turn conversational repair
- **Behavior**: Adds validation failure reasons as a new user message in the conversation
- **Best for**: Complex tasks, conversational contexts, agentic workflows
- **Context**: Builds conversation history with repair feedback
- **Requires**: ChatContext (conversational context)

**Example:**
```python
from mellea.stdlib.sampling import MultiTurnStrategy
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
result = m.instruct(
    "Write a detailed analysis...",
    requirements=[...],
    strategy=MultiTurnStrategy(loop_budget=3)
)
```

**Key Improvement**: All strategies now include detailed validation failure reasons (from `ValidationResult.reason`) when available, allowing the model to understand WHY requirements failed, not just WHICH requirements failed. This significantly improves convergence rates.

## Related Documentation

- See `mellea/stdlib/requirements/` for requirement types
- See `mellea/stdlib/sampling/` for sampling strategies
- See `docs/dev/mellea_library.md` for design philosophy
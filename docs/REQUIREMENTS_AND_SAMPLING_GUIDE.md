# Requirements and Sampling Guide

> **Complete guide to using Requirements, Sampling Strategies, and Composite Requirements in Mellea**

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements Basics](#requirements-basics)
3. [Guardrails Library](#guardrails-library)
4. [Composite Requirements](#composite-requirements)
5. [Sampling Strategies](#sampling-strategies)
6. [Advanced Patterns](#advanced-patterns)
7. [Best Practices](#best-practices)

---

## Introduction

Mellea provides a powerful system for constraining and validating LLM outputs through three key concepts:

- **Requirements**: Validation rules that check LLM outputs
- **Guardrails**: Pre-built, reusable requirements for common patterns
- **Sampling Strategies**: Algorithms that retry generation until requirements are met

This guide covers all three concepts and how to use them together effectively.

---

## Requirements Basics

### What is a Requirement?

A `Requirement` is a validation rule that checks whether an LLM output meets certain criteria. Requirements can be:

- **Simple checks**: Boolean functions that return True/False
- **LLM-based validators**: Use another LLM call to validate output
- **Hybrid validators**: Combine programmatic checks with LLM reasoning

### Creating Simple Requirements

```python
from mellea.stdlib.requirements import Requirement

# Simple boolean check
def check_length(output: str) -> bool:
    return len(output) <= 100

length_req = Requirement(
    description="Output must be 100 characters or less",
    validation_fn=check_length,
    check_only=True  # Don't provide repair feedback
)
```

### Creating LLM-Based Requirements

```python
# LLM validates the output
tone_req = Requirement(
    description="Output must have a professional tone",
    check_only=False  # Provide repair feedback when validation fails
)
# When check_only=False, the LLM provides detailed feedback on why validation failed
```

### Validation Results

Requirements return `ValidationResult` objects:

```python
from mellea.stdlib.requirements import ValidationResult

# Boolean result
result = ValidationResult(True)
assert result.as_bool() == True

# Result with reason (for repair feedback)
result = ValidationResult(
    False,
    reason="Output contains informal language like 'gonna' and 'wanna'"
)
assert result.as_bool() == False
assert "informal language" in result.reason
```

---

## Guardrails Library

The guardrails library provides 10 pre-built requirements for common validation patterns.

### Available Guardrails

#### 1. PII Detection: `no_pii()`

Detects and rejects personally identifiable information using hybrid detection (regex + spaCy NER).

```python
from mellea.stdlib.requirements.guardrails import no_pii

# Basic usage
pii_guard = no_pii()

# With custom detection mode
pii_guard = no_pii(mode="regex")  # Options: "auto", "regex", "spacy"

# Example violations:
# ❌ "Contact me at john@example.com"
# ❌ "My SSN is 123-45-6789"
# ❌ "Call me at (555) 123-4567"
# ✅ "Contact us through our website"
```

**Detection modes:**
- `"auto"` (default): Try spaCy, fallback to regex if unavailable
- `"regex"`: Fast pattern matching for emails, phones, SSNs, credit cards
- `"spacy"`: NER-based detection for names, locations, organizations

#### 2. JSON Validation: `json_valid()`

Ensures output is valid JSON.

```python
from mellea.stdlib.requirements.guardrails import json_valid

json_guard = json_valid()

# ✅ '{"name": "Alice", "age": 30}'
# ✅ '[1, 2, 3]'
# ❌ '{name: "Alice"}'  # Missing quotes
# ❌ '{"name": "Alice",}'  # Trailing comma
```

#### 3. Length Constraints: `max_length()`, `min_length()`

Enforce character or word count limits.

```python
from mellea.stdlib.requirements.guardrails import max_length, min_length

# Character limits
max_chars = max_length(100)  # Default: characters
min_chars = min_length(50)

# Word limits
max_words = max_length(20, unit="words")
min_words = min_length(10, unit="words")

# ✅ "This is a short response."  # 26 chars, 5 words
# ❌ "x" * 101  # Exceeds max_length(100)
```

#### 4. Keyword Matching: `contains_keywords()`, `excludes_keywords()`

Require or forbid specific keywords.

```python
from mellea.stdlib.requirements.guardrails import contains_keywords, excludes_keywords

# Require at least one keyword
must_have = contains_keywords(["python", "javascript", "java"])

# Require all keywords
must_have_all = contains_keywords(
    ["function", "return", "parameter"],
    require_all=True
)

# Forbid keywords
no_profanity = excludes_keywords(["damn", "hell", "crap"])

# Case sensitivity
case_sensitive = contains_keywords(["API"], case_sensitive=True)

# ✅ "Python is a great language"  # Contains "python"
# ❌ "Ruby is also nice"  # Missing required keywords
```

#### 5. Harmful Content: `no_harmful_content()`

Detects harmful, toxic, or inappropriate content using keyword-based detection.

```python
from mellea.stdlib.requirements.guardrails import no_harmful_content

# Default: checks all harm categories
safety_guard = no_harmful_content()

# Specific categories
violence_guard = no_harmful_content(harm_categories=["violence"])
profanity_guard = no_harmful_content(harm_categories=["profanity"])

# Available categories:
# - "violence": violent content
# - "hate": hate speech, discrimination
# - "sexual": sexual content
# - "self_harm": self-harm content
# - "profanity": profane language
# - "harassment": harassment, bullying

# ✅ "The weather is nice today"
# ❌ "I want to hurt someone"  # Violence
# ❌ "You're a stupid idiot"  # Harassment/profanity
```

**Future Enhancement**: See `docs/GUARDRAILS_GUARDIAN_INTEGRATION.md` for planned IBM Guardian intrinsics integration.

#### 6. JSON Schema: `matches_schema()`

Validates JSON against a JSON Schema.

```python
from mellea.stdlib.requirements.guardrails import matches_schema

# Define schema
user_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

schema_guard = matches_schema(user_schema)

# ✅ '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
# ❌ '{"name": "Bob"}'  # Missing required "age"
# ❌ '{"name": "Charlie", "age": -5}'  # Age below minimum
```

#### 7. Code Validation: `is_code()`

Validates code syntax for various programming languages.

```python
from mellea.stdlib.requirements.guardrails import is_code

# Language-specific validation
python_guard = is_code("python")
js_guard = is_code("javascript")
java_guard = is_code("java")

# Generic code validation (checks for code-like patterns)
code_guard = is_code()

# ✅ "def hello():\n    print('Hello')"  # Valid Python
# ❌ "def hello(\n    print('Hello')"  # Syntax error
# ✅ "function hello() { console.log('Hi'); }"  # Valid JS
```

**Supported languages**: `python`, `javascript`, `java`, or `None` (generic)

#### 8. Factual Grounding: `factual_grounding()`

Ensures output is grounded in provided context using token overlap.

```python
from mellea.stdlib.requirements.guardrails import factual_grounding

context = """
Python was created by Guido van Rossum and first released in 1991.
It emphasizes code readability with significant whitespace.
"""

grounding_guard = factual_grounding(context, threshold=0.3)

# ✅ "Python was created by Guido van Rossum in 1991"  # High overlap
# ❌ "Python was created by Dennis Ritchie in 1972"  # Low overlap, wrong facts
```

**Parameters:**
- `context`: Reference text for grounding
- `threshold`: Minimum token overlap ratio (default: 0.3)

**Future Enhancement**: See `docs/GUARDRAILS_NLI_GROUNDING.md` for planned NLI-based semantic grounding.

### Repair Strategies

All guardrails support repair mode (`check_only=False`) which provides detailed feedback when validation fails:

```python
from mellea.stdlib.requirements.guardrails import max_length

# Check-only mode (default)
guard_check = max_length(100, check_only=True)
# Returns: ValidationResult(False) - just pass/fail

# Repair mode
guard_repair = max_length(100, check_only=False)
# Returns: ValidationResult(False, reason="Output is 150 characters but maximum is 100. Please shorten by 50 characters.")
```

**Repair feedback examples:**

```python
# no_pii with repair
result = no_pii(check_only=False).validation_fn("Email: john@example.com", {})
# reason: "Found PII: email address (john@example.com). Please remove or redact."

# json_valid with repair
result = json_valid(check_only=False).validation_fn("{invalid}", {})
# reason: "Invalid JSON: Expecting property name enclosed in double quotes. Check syntax."

# contains_keywords with repair
result = contains_keywords(["python"], check_only=False).validation_fn("Java code", {})
# reason: "Missing required keywords: python. Please include at least one."
```

See `docs/GUARDRAILS_REPAIR_STRATEGIES.md` for complete repair strategy design.

---

## Composite Requirements

### RequirementSet

`RequirementSet` provides a composable way to combine multiple requirements:

```python
from mellea.stdlib.requirements import RequirementSet
from mellea.stdlib.requirements.guardrails import no_pii, json_valid, max_length

# Create a set
reqs = RequirementSet([
    no_pii(),
    json_valid(),
    max_length(500)
])

# Fluent API
reqs = RequirementSet().add(no_pii()).add(json_valid())

# Addition operator
reqs = RequirementSet([no_pii()]) + [json_valid(), max_length(500)]

# Extend with multiple
reqs.extend([min_length(10), contains_keywords(["data"])])
```

### GuardrailProfiles

Pre-built requirement sets for common use cases:

```python
from mellea.stdlib.requirements import GuardrailProfiles

# 1. Basic Safety - PII + harmful content
reqs = GuardrailProfiles.basic_safety()

# 2. JSON Output - Valid JSON with length limits
reqs = GuardrailProfiles.json_output(max_length=1000)

# 3. Code Generation - Valid code with safety
reqs = GuardrailProfiles.code_generation(language="python")

# 4. Professional Content - Safe, appropriate, length-limited
reqs = GuardrailProfiles.professional_content(max_length=500)

# 5. API Documentation - Code + JSON + professional
reqs = GuardrailProfiles.api_documentation(language="python")

# 6. Grounded Summary - Factually grounded with length limits
reqs = GuardrailProfiles.grounded_summary(context="...", max_length=200)

# 7. Safe Chat - Conversational safety
reqs = GuardrailProfiles.safe_chat(max_length=300)

# 8. Structured Data - JSON with optional schema
reqs = GuardrailProfiles.structured_data(schema={...})

# 9. Content Moderation - Comprehensive safety
reqs = GuardrailProfiles.content_moderation()

# 10. Minimal - Just PII detection
reqs = GuardrailProfiles.minimal()

# 11. Strict - All safety + format + length
reqs = GuardrailProfiles.strict(max_length=500)
```

### Customizing Profiles

```python
# Start with a profile and customize
reqs = GuardrailProfiles.safe_chat()
reqs = reqs.add(matches_schema(my_schema))
reqs = reqs.remove(max_length(300))  # Remove default length limit
reqs = reqs.add(max_length(1000))    # Add custom limit

# Compose multiple profiles
reqs = GuardrailProfiles.basic_safety() + GuardrailProfiles.json_output()
```

---

## Sampling Strategies

Sampling strategies control how Mellea retries generation when requirements fail.

### Available Strategies

#### 1. RejectionSamplingStrategy

Simplest strategy: retry the same prompt until requirements pass or budget exhausted.

```python
from mellea.stdlib.sampling import RejectionSamplingStrategy

strategy = RejectionSamplingStrategy(
    loop_budget=3,  # Try up to 3 times
    requirements=[no_pii(), json_valid()]
)

# Use with session
result = await session.generate(
    instruction,
    sampling_strategy=strategy
)
```

**How it works:**
1. Generate output
2. Validate against requirements
3. If failed, retry with same prompt
4. Repeat until success or budget exhausted

**Best for:** Simple retries without prompt modification

#### 2. RepairTemplateStrategy

Adds failure feedback to the prompt on retry.

```python
from mellea.stdlib.sampling import RepairTemplateStrategy

strategy = RepairTemplateStrategy(
    loop_budget=3,
    requirements=[no_pii(), max_length(100)]
)
```

**How it works:**
1. Generate output
2. Validate against requirements
3. If failed, add failure details to prompt:
   ```
   The following requirements failed before:
   * Output contains PII: email address
   * Output exceeds maximum length of 100 characters
   ```
4. Retry with enhanced prompt

**Best for:** Giving the model feedback to improve

#### 3. MultiTurnStrategy

Uses multi-turn conversation for repair (agentic approach).

```python
from mellea.stdlib.sampling import MultiTurnStrategy

strategy = MultiTurnStrategy(
    loop_budget=3,
    requirements=[json_valid(), contains_keywords(["summary"])]
)
```

**How it works:**
1. Generate output
2. Validate against requirements
3. If failed, add user message:
   ```
   The following requirements have not been met:
   * Output is not valid JSON
   * Output must contain keyword: summary
   Please try again to fulfill the requirements.
   ```
4. Model responds in conversation

**Best for:** Complex tasks where conversation helps

**Requires:** `ChatContext` (not compatible with simple `Context`)

#### 4. MajorityVotingStrategyForMath

Generates multiple samples and selects best via majority voting (MBRD).

```python
from mellea.stdlib.sampling import MajorityVotingStrategyForMath

strategy = MajorityVotingStrategyForMath(
    number_of_samples=8,  # Generate 8 candidates
    loop_budget=1,
    requirements=[contains_keywords(["answer"])]
)
```

**How it works:**
1. Generate N samples concurrently
2. Compare all pairs using math expression equivalence
3. Select sample with highest agreement score

**Best for:** Math problems where multiple solutions should agree

#### 5. MBRDRougeLStrategy

Like majority voting but uses RougeL for text similarity.

```python
from mellea.stdlib.sampling import MBRDRougeLStrategy

strategy = MBRDRougeLStrategy(
    number_of_samples=5,
    loop_budget=1,
    requirements=[min_length(50)]
)
```

**Best for:** Text generation where consistency matters

#### 6. BudgetForcingSamplingStrategy

Allocates token budget for thinking vs. answering (Ollama only).

```python
from mellea.stdlib.sampling import BudgetForcingSamplingStrategy

strategy = BudgetForcingSamplingStrategy(
    think_max_tokens=4096,  # Tokens for reasoning
    answer_max_tokens=512,  # Tokens for final answer
    start_think_token="<think>",
    end_think_token="</think>",
    loop_budget=1,
    requirements=[is_code("python")]
)
```

**How it works:**
1. Model generates reasoning in `<think>` block
2. Forces model to continue thinking if needed
3. Generates final answer with separate token budget

**Best for:** Complex reasoning tasks (Ollama backend only)

### Choosing a Strategy

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **RejectionSampling** | Simple retries | Fast, simple | No learning from failures |
| **RepairTemplate** | Iterative improvement | Model learns from errors | Requires good repair feedback |
| **MultiTurn** | Complex tasks | Conversational repair | Requires ChatContext |
| **MajorityVoting** | Math/reasoning | Robust to errors | Expensive (N samples) |
| **MBRDRougeL** | Text consistency | Good for summaries | Expensive (N samples) |
| **BudgetForcing** | Deep reasoning | Explicit thinking | Ollama only |

---

## Advanced Patterns

### Pattern 1: Progressive Validation

Start with cheap checks, add expensive ones only if needed:

```python
# Fast checks first
fast_reqs = RequirementSet([
    json_valid(),
    max_length(1000)
])

# Expensive checks later
expensive_reqs = RequirementSet([
    matches_schema(complex_schema),
    factual_grounding(long_context)
])

# Use in stages
result = await session.generate(
    instruction,
    sampling_strategy=RejectionSamplingStrategy(
        loop_budget=2,
        requirements=fast_reqs.to_list()
    )
)

# Only validate expensive if fast checks passed
if result.success:
    # Validate with expensive checks
    ...
```

### Pattern 2: Conditional Requirements

Apply different requirements based on task type:

```python
def get_requirements(task_type: str) -> list[Requirement]:
    base = [no_pii(), no_harmful_content()]
    
    if task_type == "code":
        return base + [is_code("python"), max_length(2000)]
    elif task_type == "summary":
        return base + [min_length(50), max_length(200)]
    elif task_type == "json":
        return base + [json_valid(), matches_schema(schema)]
    else:
        return base

reqs = get_requirements("code")
```

### Pattern 3: Layered Sampling

Combine multiple strategies:

```python
# First: Try with repair feedback
result = await session.generate(
    instruction,
    sampling_strategy=RepairTemplateStrategy(
        loop_budget=2,
        requirements=[json_valid(), no_pii()]
    )
)

# If failed: Try majority voting
if not result.success:
    result = await session.generate(
        instruction,
        sampling_strategy=MBRDRougeLStrategy(
            number_of_samples=5,
            requirements=[json_valid(), no_pii()]
        )
    )
```

### Pattern 4: Custom Validation Functions

Create domain-specific requirements:

```python
def validate_email_format(output: str, context: dict) -> ValidationResult:
    """Validate email has proper format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, output.strip()):
        return ValidationResult(True)
    else:
        return ValidationResult(
            False,
            reason="Email format invalid. Expected: user@domain.com"
        )

email_req = Requirement(
    description="Output must be a valid email address",
    validation_fn=validate_email_format,
    check_only=False
)
```

### Pattern 5: Dynamic Requirements

Generate requirements based on runtime data:

```python
def create_keyword_requirements(keywords: list[str]) -> RequirementSet:
    """Create requirements for multiple keyword sets."""
    reqs = RequirementSet()
    
    for keyword in keywords:
        reqs = reqs.add(contains_keywords([keyword]))
    
    return reqs

# Use with dynamic data
user_keywords = ["python", "async", "await"]
reqs = create_keyword_requirements(user_keywords)
```

---

## Best Practices

### 1. Start Simple, Add Complexity

```python
# ❌ Don't start with everything
reqs = GuardrailProfiles.strict(max_length=100) + [
    matches_schema(complex_schema),
    factual_grounding(huge_context),
    is_code("python")
]

# ✅ Start minimal, add as needed
reqs = GuardrailProfiles.minimal()  # Just PII
# Test, then add more...
reqs = reqs.add(json_valid())
# Test, then add more...
reqs = reqs.add(max_length(500))
```

### 2. Use Repair Mode for Development

```python
# During development: get detailed feedback
dev_reqs = RequirementSet([
    no_pii(check_only=False),
    json_valid(check_only=False),
    max_length(100, check_only=False)
])

# In production: faster check-only mode
prod_reqs = RequirementSet([
    no_pii(check_only=True),
    json_valid(check_only=True),
    max_length(100, check_only=True)
])
```

### 3. Profile Before Optimizing

```python
import time

start = time.time()
result = await session.generate(
    instruction,
    sampling_strategy=strategy,
    requirements=reqs
)
elapsed = time.time() - start

print(f"Generation took {elapsed:.2f}s")
print(f"Attempts: {len(result.sample_generations)}")
print(f"Success: {result.success}")
```

### 4. Handle Failures Gracefully

```python
result = await session.generate(
    instruction,
    sampling_strategy=RepairTemplateStrategy(loop_budget=3),
    requirements=reqs
)

if result.success:
    print("✅ All requirements met")
    output = result.result
else:
    print("⚠️ Requirements not met after 3 attempts")
    # Use best attempt
    output = result.result
    
    # Log failures for analysis
    for i, validation in enumerate(result.sample_validations):
        failed = [r for r, v in validation if not v.as_bool()]
        print(f"Attempt {i+1} failed: {len(failed)} requirements")
```

### 5. Combine Profiles Wisely

```python
# ✅ Good: Complementary profiles
reqs = GuardrailProfiles.basic_safety() + GuardrailProfiles.json_output()

# ❌ Bad: Conflicting requirements
reqs = GuardrailProfiles.minimal() + GuardrailProfiles.strict()
# (strict already includes minimal, creates duplicates)

# ✅ Better: Use strict directly
reqs = GuardrailProfiles.strict()
```

### 6. Test Requirements Independently

```python
# Test each requirement separately
test_output = '{"name": "test@example.com"}'

for req in reqs:
    result = req.validation_fn(test_output, {})
    print(f"{req.description}: {result.as_bool()}")
    if not result.as_bool() and result.reason:
        print(f"  Reason: {result.reason}")
```

### 7. Use Type Hints

```python
from mellea.stdlib.requirements import Requirement, RequirementSet, ValidationResult

def create_requirements() -> RequirementSet:
    """Create requirements with proper typing."""
    return RequirementSet([
        no_pii(),
        json_valid()
    ])

def custom_validator(output: str, context: dict) -> ValidationResult:
    """Custom validator with proper return type."""
    return ValidationResult(len(output) > 0)
```

### 8. Document Your Requirements

```python
# ✅ Good: Clear descriptions
email_req = Requirement(
    description="Output must be a valid email in format user@domain.com",
    validation_fn=validate_email,
    check_only=False
)

# ❌ Bad: Vague descriptions
email_req = Requirement(
    description="Check email",
    validation_fn=validate_email
)
```

---

## Summary

**Requirements** validate LLM outputs against rules:
- Use **guardrails** for common patterns (PII, JSON, length, etc.)
- Create **custom requirements** for domain-specific needs
- Enable **repair mode** for detailed feedback

**Composite Requirements** organize validation:
- Use **RequirementSet** for flexible composition
- Use **GuardrailProfiles** for pre-built combinations
- Customize profiles for your use case

**Sampling Strategies** handle retries:
- **RejectionSampling**: Simple retries
- **RepairTemplate**: Learning from failures
- **MultiTurn**: Conversational repair
- **MajorityVoting**: Consensus from multiple samples
- **BudgetForcing**: Explicit reasoning (Ollama)

**Best Practices:**
1. Start simple, add complexity gradually
2. Use repair mode during development
3. Profile performance before optimizing
4. Handle failures gracefully
5. Test requirements independently
6. Document clearly

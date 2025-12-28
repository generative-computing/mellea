# KG-RAG Refactoring Guide: Embracing Mellea's Patterns

This guide explains how the KG-RAG implementation has been refactored to align with Mellea's philosophy and design patterns.

## Overview

The refactored version transforms KG-RAG from **"using Mellea as a backend"** to **"building with Mellea's composable patterns"**. This makes the code more maintainable, type-safe, and aligned with functional programming principles.

## 📚 Refactoring Documentation

This repository includes two comprehensive refactoring guides:

1. **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** (this file) - Covers the main KG-RAG components:
   - `@generative` decorator for LLM functions
   - Pydantic models for structured outputs
   - Requirements for validation
   - KGRagComponent architecture

2. **[PREPROCESSOR_REFACTORING.md](PREPROCESSOR_REFACTORING.md)** - Covers the KG preprocessing pipeline:
   - Refactored preprocessor with type safety
   - Neo4j connection management
   - Batch operations for performance
   - Configuration management

## Key Changes

### 1. Pydantic Models for Structured Outputs

**Before** ([kg_model.py](kg_model.py)):
```python
response = await generate_response(...)
routes = maybe_load_json(response)["routes"]  # Manual JSON parsing, error-prone
```

**After** ([kg_models.py](kg_models.py)):
```python
from pydantic import BaseModel, Field

class QuestionRoutes(BaseModel):
    reason: str = Field(description="Reasoning for the route ordering")
    routes: List[List[str]] = Field(description="List of solving routes")

# Now type-safe and validated automatically
result: QuestionRoutes = await break_down_question(...)
```

**Benefits:**
- Type safety with IDE autocomplete
- Automatic validation of LLM outputs
- Clear data contracts between functions
- Better error messages when parsing fails

### 2. @generative Decorator for LLM Functions

**Before**:
```python
@llm_retry(max_retries=MAX_RETRIES, default_output=[])
async def break_down_question(self, query: Query) -> List[Query]:
    system_prompt = PROMPTS["break_down_question"]["system"].format(...)
    user_message = PROMPTS["break_down_question"]["user"].format(...)

    response = await generate_response(
        self.session,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_message}],
        max_tokens=2048,
        response_format={"type": "json_object"},
        logger=self.logger
    )
    routes = maybe_load_json(response)["routes"]
    # ... manual conversion logic
```

**After** ([kg_generative.py](kg_generative.py)):
```python
from mellea.stdlib.genslot import generative

@generative
async def break_down_question(
    query: str,
    query_time: str,
    domain: str,
    route: int,
    hints: str
) -> QuestionRoutes:
    """Break down a complex question into multiple solving routes.

    You are a helpful assistant who is good at answering questions in the {domain} domain...

    Question: {query}
    Query Time: {query_time}
    """
    pass
```

**Benefits:**
- Prompts live with the function (better locality)
- Automatic JSON parsing via Pydantic return type
- No manual session management
- Cleaner, more declarative code
- Easy to test in isolation

### 3. Requirements Instead of Custom Retry Logic

**Before** ([utils/utils.py:47-73](utils/utils.py#L47-L73)):
```python
def llm_retry(max_retries=10, default_output=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except json.JSONDecodeError:
                    logger.error(f"[Retry {attempt+1}/{max_retries}] JSON Decode error")
                    await asyncio.sleep(min(2 ** attempt, 30))
                # ... more error handling
            return default_output
        return wrapper
    return decorator
```

**After** ([kg_requirements.py](kg_requirements.py)):
```python
from mellea.stdlib.requirement import Requirement

VALID_JSON_REQ = Requirement(
    name="valid_json",
    requirement="Output must be valid JSON format",
    validator=is_valid_json
)

ROUTES_PRESENT_REQ = Requirement(
    name="routes_present",
    requirement="Output must contain 'routes' field with at least one route",
    validator=has_nonempty_list("routes")
)

# Use with RejectionSampling
from mellea.stdlib.sampling.rejection_sampling import RejectionSamplingStrategy

result = await session.instruct(
    instruction=prompt,
    requirements=[VALID_JSON_REQ, ROUTES_PRESENT_REQ],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

**Benefits:**
- Declarative validation constraints
- Built-in retry logic with Mellea's sampling strategies
- Composable validators
- Better separation of concerns
- Integrated with Mellea's validation framework

### 4. Jinja2 Templates for Prompts

**Before**: Prompts scattered in [kg_model.py:33-452](kg_model.py#L33-L452) as Python strings

**After**: Organized template files in [templates/](templates/)

```
templates/
├── break_down_question.jinja2
├── extract_entity.jinja2
└── evaluate.jinja2
```

**Example template** ([templates/break_down_question.jinja2](templates/break_down_question.jinja2)):
```jinja2
You are a helpful assistant who is good at answering questions in the {{ domain }} domain...

Domain-specific Hints:
{{ hints }}

Question: {{ query }}
Query Time: {{ query_time }}
```

**Benefits:**
- Separation of prompts from code
- Reusable and version-controlled
- Easier to iterate on prompts
- Standard Mellea pattern

### 5. KGRagComponent Architecture

**Before**: `KGModel` class with mixed concerns

**After** ([kg_rag_refactored.py](kg_rag_refactored.py)):
```python
from mellea.stdlib.base import Component

class KGRagComponent(Component):
    """Knowledge Graph-Enhanced RAG component using Mellea patterns."""

    def __init__(self, session, eval_session, emb_session, domain, config, logger):
        super().__init__()
        # ... initialization

    async def execute(self, query: str, query_time: datetime, ...) -> str:
        """Execute KG-RAG pipeline to answer a query."""
        # Composable with other Mellea components
        pass
```

**Benefits:**
- Proper Mellea Component pattern
- Composable with other components
- Clear interface via `execute()` method
- Can be used in larger Mellea pipelines

### 6. Simplified Utils with Mellea Features

**Before** ([utils/utils.py:128-162](utils/utils.py#L128-L162)): Manual prompt management, tokenization, truncation

**After** ([kg_utils_mellea.py](kg_utils_mellea.py)):
```python
from mellea.helpers.logging import logger

async def generate_embedding_mellea(session, texts, **kwargs):
    """Generate embeddings using Mellea session or local model."""
    # Simplified logic, leveraging Mellea's abstractions
    if hasattr(session, "embeddings"):
        responses = await session.embeddings.create(input=texts, **kwargs)
        return [data.embedding for data in responses.data]
    elif hasattr(session, "encode"):
        return session.encode(sentences=texts, normalize_embeddings=True).tolist()
```

**Benefits:**
- Uses Mellea's logging
- Simpler, cleaner code
- Less custom infrastructure
- Leverages Mellea's context management

## File Structure Comparison

### Before
```
docs/examples/kgrag/
├── kg_model.py           # 1150+ lines, mixed concerns
├── utils/utils.py        # Custom retry, tokenization, etc.
├── constants.py
├── eval.py
└── run/
    ├── run_qa.py
    └── ...
```

### After
```
docs/examples/kgrag/
├── kg_models.py          # Pydantic models (60 lines)
├── kg_generative.py      # @generative functions (200 lines)
├── kg_requirements.py    # Mellea Requirements (100 lines)
├── kg_rag_refactored.py  # KGRagComponent (500 lines)
├── kg_utils_mellea.py    # Simplified utils (70 lines)
├── demo_refactored.py    # Simple demo script
├── templates/            # Jinja2 prompt templates
│   ├── break_down_question.jinja2
│   ├── extract_entity.jinja2
│   └── evaluate.jinja2
└── kg_model.py          # Original (for comparison)
```

## Usage Comparison

### Before
```python
from kg_model import KGModel

kg_model = KGModel(
    session=session,
    eval_session=eval_session,
    emb_session=emb_session,
    domain="movie",
    config={"route": 5, "width": 30, "depth": 3}
)

answer = await kg_model.generate_answer(
    query="Who won best actor in 2020?",
    query_time=datetime(2024, 3, 19)
)
```

### After
```python
from kg_rag_refactored import KGRagComponent

kg_rag = KGRagComponent(
    session=session,
    eval_session=eval_session,
    emb_session=emb_session,
    domain="movie",
    config={"route": 5, "width": 30, "depth": 3}
)

# Cleaner interface
answer = await kg_rag.execute(
    query="Who won best actor in 2020?",
    query_time=datetime(2024, 3, 19)
)

# Can also use compositionally
result = await session.instruct(
    instruction="Answer using knowledge graph",
    components=[kg_rag],
    user_variables={"query": "..."}
)
```

## Key Philosophical Shifts

### 1. From Imperative to Declarative
- **Before**: Manually call `generate_response()`, parse JSON, handle errors
- **After**: Declare what you want with `@generative` and Pydantic models

### 2. From Custom Infrastructure to Mellea Patterns
- **Before**: Custom `@llm_retry`, `Token_Counter`, `generate_response()`
- **After**: Mellea's Requirements, SamplingStrategies, built-in token tracking

### 3. From Monolithic to Composable
- **Before**: Single `KGModel` class with all logic
- **After**: Composable functions and components that follow Mellea patterns

### 4. From Strings to Types
- **Before**: String-based prompts, manual JSON parsing
- **After**: Type-safe Pydantic models, automatic validation

### 5. From Mixed Concerns to Separation
- **Before**: Prompts, logic, validation all mixed
- **After**: Clear separation: templates, models, requirements, components

## Running the Refactored Version

```bash
# Set up environment (same as before)
cd docs/examples/kgrag
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run demo
uv run --with mellea python demo_refactored.py
```

## Migration Path

For existing users, you can gradually migrate:

1. **Keep using `kg_model.py`** - Original still works
2. **Try `demo_refactored.py`** - See the new patterns in action
3. **Adopt incrementally**:
   - Start with Pydantic models for type safety
   - Replace `@llm_retry` with Requirements
   - Refactor one function at a time to use `@generative`
   - Move to `KGRagComponent` when ready

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Type Safety** | Manual JSON parsing | Pydantic models |
| **Validation** | Custom retry decorator | Mellea Requirements |
| **Prompts** | Python strings | Jinja2 templates |
| **LLM Calls** | Manual session management | @generative decorator |
| **Components** | Monolithic class | Composable Component |
| **Testing** | Hard to test | Easy to test isolated functions |
| **Maintenance** | 1150+ line class | Modular, ~100 lines per file |
| **Composability** | Limited | Full Mellea integration |

## Next Steps

1. **Add more requirements** - Domain-specific validators
2. **Use Majority Voting** - Replace manual consensus with `MajorityVotingStrategy`
3. **LLM-as-judge evaluation** - Use `mellea.stdlib.test_based_eval`
4. **Extend to other domains** - Finance, scientific literature, etc.
5. **Add caching** - Leverage Mellea's context caching

## Further Reading

- [Mellea Documentation](../../README.md)
- [CLAUDE.md](../../../CLAUDE.md) - Project-specific guidance
- [Original KG-RAG README](README.md)
- [Bidirection Paper](https://github.com/junhongmit/Bidirection)

## Questions?

The refactored version demonstrates Mellea's philosophy of building composable, type-safe generative programs. Both versions (original and refactored) coexist in this repository for comparison and gradual migration.

For questions or suggestions, please open an issue on GitHub.

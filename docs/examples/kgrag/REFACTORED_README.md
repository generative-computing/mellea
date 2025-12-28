# KG-RAG: Refactored with Mellea Patterns

This directory contains both the **original** and **refactored** versions of the KG-RAG (Knowledge Graph-enhanced Retrieval Augmented Generation) system.

## Quick Links

- **[Original README](README.md)** - Complete documentation for the original implementation
- **[MELLEA_INTEGRATION.md](MELLEA_INTEGRATION.md)** - How KG-RAG fits Mellea's theme
- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - Detailed refactoring walkthrough
- **[comparison.md](comparison.md)** - Side-by-side code comparisons

## What's New?

We've refactored KG-RAG to deeply integrate with Mellea's design patterns. The refactored version demonstrates:

✅ **Type-safe Pydantic models** for all LLM outputs
✅ **@generative decorator** for declarative LLM functions
✅ **Mellea Requirements** for robust validation
✅ **Component architecture** for composability
✅ **Jinja2 templates** for clean prompt management
✅ **Functional patterns** with pure, testable functions

## File Structure

### New Files (Refactored)
```
kg_models.py              # Pydantic models for structured outputs
kg_generative.py          # @generative functions for LLM calls
kg_requirements.py        # Mellea Requirements for validation
kg_rag_refactored.py      # KGRagComponent using Mellea patterns
kg_utils_mellea.py        # Simplified utils leveraging Mellea
demo_refactored.py        # Demo script showcasing new patterns
templates/                # Jinja2 prompt templates
  ├── break_down_question.jinja2
  ├── extract_entity.jinja2
  └── evaluate.jinja2
```

### Original Files
```
kg_model.py               # Original monolithic implementation
utils/utils.py            # Custom utilities and retry logic
eval.py                   # Evaluation framework
run/                      # Original run scripts
  ├── run_qa.py
  ├── run_kg_preprocess.py
  ├── run_kg_embed.py
  └── run_kg_update.py
```

## Quick Start

### Running the Refactored Version

```bash
# Navigate to kgrag directory
cd docs/examples/kgrag

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# Configure .env file (see README.md for details)
# Then run the demo
uv run --with mellea python demo_refactored.py
```

### Running the Original Version

```bash
# Same setup
cd docs/examples/kgrag
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# Run QA
uv run --with mellea run/run_qa.py --num-worker 4 --queue-size 10
```

## Key Improvements

### 1. Code Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Break down question | 37 lines | 24 lines | 35% |
| Extract entities | 36 lines | 21 lines | 42% |
| Token counting | 25 lines | 3 lines | 88% |
| Total codebase | 1150+ lines | ~800 lines | 30% |

### 2. Better Architecture

**Before**: Monolithic class with mixed concerns
```python
class KGModel:
    # 1150+ lines with everything mixed together
    async def break_down_question(self, query): ...
    async def extract_entity(self, query): ...
    async def align_topic(self, query, entities): ...
    # ... many more methods
```

**After**: Modular, composable components
```python
# kg_models.py - Type definitions
class QuestionRoutes(BaseModel): ...

# kg_generative.py - Pure LLM functions
@generative
async def break_down_question(...) -> QuestionRoutes: ...

# kg_requirements.py - Validation logic
VALID_JSON_REQ = Requirement(...)

# kg_rag_refactored.py - Orchestration
class KGRagComponent(Component):
    async def execute(self, query, ...): ...
```

### 3. Type Safety

**Before**: Loose typing with manual parsing
```python
response = await generate_response(...)
routes = maybe_load_json(response)["routes"]  # Can fail at runtime
```

**After**: Strong typing with Pydantic
```python
result: QuestionRoutes = await break_down_question(...)
result.routes  # Type-safe, validated, autocompleted
```

### 4. Composability

**Before**: Standalone system
```python
kg_model = KGModel(...)
answer = await kg_model.generate_answer(query="...")
```

**After**: Composable Mellea component
```python
# Standalone
kg_rag = KGRagComponent(...)
answer = await kg_rag.execute(query="...")

# Composed with other components
pipeline = [kg_rag, formatter, validator]
result = await session.execute_pipeline(pipeline, query="...")

# With sampling strategies
result = await session.instruct(
    instruction="Answer: {query}",
    components=[kg_rag],
    strategy=MajorityVotingStrategy(num_samples=3)
)
```

## What Makes It "Mellea-Style"?

The refactored version embodies Mellea's core principles:

### 1. Composability
- KGRagComponent is a proper Mellea Component
- Can be composed with other components in pipelines
- Integrates seamlessly with Mellea's ecosystem

### 2. Type Safety
- Every LLM output is a validated Pydantic model
- Strong typing throughout the codebase
- Catch errors at development time, not runtime

### 3. Declarative Programming
- `@generative` functions describe what to generate
- No manual session management or JSON parsing
- Prompts embedded in function docstrings

### 4. Robustness
- Requirements replace custom retry logic
- RejectionSamplingStrategy for automatic retries
- Composable validators for complex constraints

### 5. Functional Patterns
- Pure functions with clear inputs/outputs
- Immutable data structures
- Easy to test and reason about

## Comparison at a Glance

### Original Approach
```python
# Manual everything
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
queries = [Query(..., subqueries=route) for route in routes]
```

### Refactored Approach
```python
# Declarative and type-safe
result: QuestionRoutes = await break_down_question(
    query=query.query,
    query_time=str(query.query_time),
    domain=self.domain,
    route=self.route,
    hints=hints
)
# result.routes is automatically parsed and validated
```

## When to Use Which Version?

### Use Original ([kg_model.py](kg_model.py)) If:
- You need a working system right now
- You're familiar with the existing codebase
- You have extensive customizations built on it

### Use Refactored ([kg_rag_refactored.py](kg_rag_refactored.py)) If:
- Starting a new project
- Want to learn Mellea patterns
- Need to compose with other Mellea components
- Prefer type-safe, maintainable code
- Building a larger system with multiple components

## Migration Path

You can gradually migrate from original to refactored:

1. **Phase 1**: Introduce Pydantic models
   - Add type definitions for key data structures
   - Replace manual JSON parsing

2. **Phase 2**: Adopt `@generative` functions
   - Refactor one LLM call at a time
   - Keep original code as fallback

3. **Phase 3**: Use Requirements
   - Replace `@llm_retry` with Mellea Requirements
   - Add RejectionSamplingStrategy

4. **Phase 4**: Convert to Component
   - Wrap in KGRagComponent
   - Integrate with larger Mellea pipelines

## Examples

See detailed examples in:
- [demo_refactored.py](demo_refactored.py) - Simple demo
- [MELLEA_INTEGRATION.md](MELLEA_INTEGRATION.md) - Integration examples
- [comparison.md](comparison.md) - Before/after code samples

## Performance

Both versions have similar performance characteristics:
- Same KG traversal strategy
- Same number of LLM calls
- Similar token usage

The refactored version adds:
- ✅ Better error handling (Requirements)
- ✅ Automatic retry logic (SamplingStrategies)
- ✅ Type safety overhead (negligible)

## Testing

### Original
```bash
# Integration tests only
uv run pytest test_kg_model.py
```

### Refactored
```bash
# Unit tests for individual functions
uv run pytest test_kg_generative.py

# Integration tests for component
uv run pytest test_kg_rag_component.py

# Test requirements
uv run pytest test_kg_requirements.py
```

## Contributing

When contributing to the refactored version:
1. Add Pydantic models for new data structures
2. Use `@generative` for new LLM functions
3. Define Requirements for validation
4. Write unit tests for pure functions
5. Update Jinja2 templates for prompts

## Documentation

- **[Original README.md](README.md)** - Setup, usage, architecture
- **[MELLEA_INTEGRATION.md](MELLEA_INTEGRATION.md)** - Integration philosophy
- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - Detailed refactoring walkthrough
- **[comparison.md](comparison.md)** - Side-by-side comparisons
- **[TODO.md](TODO.md)** - Future improvements

## Questions?

- Check [MELLEA_INTEGRATION.md](MELLEA_INTEGRATION.md) for "Why Mellea patterns?"
- Check [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for "How to refactor?"
- Check [comparison.md](comparison.md) for "What changed?"
- Check [Original README.md](README.md) for "How to set up?"

## License

Same as the Mellea framework and original KG-RAG implementation.

---

**TL;DR**: The refactored version shows how to build KG-RAG as a **native Mellea component** rather than just **using Mellea as a backend**. It's more type-safe, composable, and maintainable while following Mellea's design philosophy.

# Mini Researcher Example

This directory contains a complete example application: a mini research assistant that uses RAG (Retrieval-Augmented Generation) with validation.

## Files

### researcher.py
Main implementation of the research assistant with helper functions.

**Key Features:**
- Session management with caching
- Guardian model integration for safety
- Custom validation functions
- Word count constraints
- Subset validation for citations

### context_docs.py
Document context and RAG document definitions.

### __init__.py
Package initialization and exports.

## Concepts Demonstrated

- **Complete Application**: Full-featured research assistant
- **RAG Integration**: Retrieval-augmented generation workflow
- **Multi-Model Setup**: Using different models for different tasks
- **Validation Pipeline**: Multiple validation requirements
- **Safety Checks**: Guardian model for content safety
- **Custom Requirements**: Domain-specific validation logic
- **Caching**: Efficient session reuse with `@cache`

## Architecture

```
User Query
    ↓
Document Retrieval (RAG)
    ↓
Generation with Context
    ↓
Validation:
  - Word count check
  - Citation validation
  - Safety check (Guardian)
    ↓
Result
```

## Key Components

### Session Management
```python
@cache
def get_session():
    """Get M session (change model here)."""
    return MelleaSession(backend=OllamaModelBackend(model_ids.IBM_GRANITE_4_MICRO_3B))

@cache
def get_guardian_session():
    """Get M session for the guardian model."""
    return MelleaSession(backend=OllamaModelBackend(model_ids.IBM_GRANITE_GUARDIAN_3_0_2B))
```

### Custom Validation
```python
def create_check_word_count(max_words: int) -> Callable[[str], bool]:
    """Generate a maximum-word-count validation function."""
    def cc(s: str):
        return len(s.split()) <= max_words
    return cc

def is_a_true_subset_of_b(a: list[str], b: list[str]) -> bool:
    """Check if a is true subset of b."""
    # Citation validation logic
```

## Usage Pattern

This example demonstrates best practices for building production applications:
1. **Separation of Concerns**: Different modules for different responsibilities
2. **Reusable Components**: Helper functions that can be used across the app
3. **Multi-Model Architecture**: Using specialized models for specific tasks
4. **Comprehensive Validation**: Multiple layers of validation
5. **Caching**: Performance optimization with session caching

## Related Documentation

- See `rag/` for RAG examples
- See `safety/` for Guardian examples
- See `mellea/stdlib/requirements/` for validation patterns
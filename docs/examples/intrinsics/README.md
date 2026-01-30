# Intrinsics Examples

This directory contains examples for using Mellea's intrinsic functions - specialized model capabilities accessed through adapters.

## Files

### intrinsics.py
Core example showing how to directly use intrinsics with adapters.

**Key Features:**
- Creating and adding adapters to backends
- Using `Intrinsic` component for specialized tasks
- Working with Granite Common adapters (aLoRA-based)
- Understanding adapter output formats

### answer_relevance.py
Evaluates whether an answer is relevant to a question.

### answerability.py
Checks if a question can be answered given the context.

### citations.py
Validates and extracts citations from generated text.

### context_relevance.py
Assesses if retrieved context is relevant to a query.

### hallucination_detection.py
Detects when model outputs contain hallucinated information.

### query_rewrite.py
Rewrites queries for better retrieval or understanding.

## Concepts Demonstrated

- **Intrinsic Functions**: Specialized model capabilities beyond text generation
- **Adapter System**: Using LoRA/aLoRA adapters for specific tasks
- **RAG Evaluation**: Assessing retrieval-augmented generation quality
- **Quality Metrics**: Measuring relevance, groundedness, and accuracy
- **Backend Integration**: Adding adapters to different backend types

## Basic Usage

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters.adapter import GraniteCommonAdapter
from mellea.stdlib.components import Intrinsic
import mellea.stdlib.functional as mfuncs

# Create backend and adapter
backend = LocalHFBackend(model_id="ibm-granite/granite-3.3-8b-instruct")
adapter = GraniteCommonAdapter("requirement_check", 
                               base_model_name=backend.base_model_name)
backend.add_adapter(adapter)

# Use intrinsic
out, new_ctx = mfuncs.act(
    Intrinsic("requirement_check",
              intrinsic_kwargs={"requirement": "The assistant is helpful."}),
    ctx,
    backend
)
```

## Available Intrinsics

- **requirement_check**: Validate requirements (used by ALoraRequirement)
- **answer_relevance**: Check answer-question relevance
- **answerability**: Determine if question is answerable
- **citations**: Extract and validate citations
- **context_relevance**: Assess context-query relevance
- **hallucination_detection**: Detect hallucinated content
- **query_rewrite**: Improve query formulation

## Related Documentation

- See `mellea/stdlib/components/intrinsic/` for intrinsic implementations
- See `mellea/backends/adapters/` for adapter system
- See `docs/dev/intrinsics_and_adapters.md` for architecture details
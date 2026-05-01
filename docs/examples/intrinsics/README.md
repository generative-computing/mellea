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

### uncertainty.py
Estimates the model's certainty about answering a question.

### requirement_check.py
Detect if text adheres to provided requirements.

### policy_guardrails.py
Checks if a scenario is compliant/non-compliant/ambiguous with respect to a given policy,

### guardian_core.py
Uses the guardian-core LoRA adapter for safety risk detection, including prompt-level harm, response-level social bias, RAG groundedness, and custom criteria.

### factuality_detection.py
Detects if the the model's output is factually incorrect relative to context.

### factuality_correction.py
Corrects a factually incorrect response relative to context.

### context_attribution.py
Identifies sentences in conversation history and documents that most influenced the response.


## Concepts Demonstrated

- **Intrinsic Functions**: Specialized model capabilities beyond text generation
- **Adapter System**: Using LoRA/aLoRA adapters for specific tasks
- **RAG Evaluation**: Assessing retrieval-augmented generation quality
- **Quality Metrics**: Measuring relevance, groundedness, and accuracy
- **Backend Integration**: Adding adapters to different backend types (LocalHFBackend with runtime adapters, OpenAIBackend with Granite Switch embedded adapters)

## Basic Usage

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters.adapter import IntrinsicAdapter
from mellea.stdlib.components import Intrinsic
import mellea.stdlib.functional as mfuncs

# Create backend and adapter
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
adapter = IntrinsicAdapter("requirement_check",
                               base_model_name=backend.base_model_name)
backend.add_adapter(adapter)

# Use intrinsic
out, new_ctx = mfuncs.act(
    Intrinsic(
        "requirement_check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."}),
    ctx,
    backend
)
```

OpenAIBackends also support a type of embedded adapter for Granite Switch models:
```python
backend = OpenAIBackend(
        model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
        load_embedded_adapters=True,  # Auto-loads adapters from huggingface repo.
        ...
)

# Assumes the model has the `requirement_check` adapter embedded.
out, new_ctx = mfuncs.act(
    Intrinsic(
        "requirement_check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."}),
    ctx,
    backend
)
```

For complete runnable examples using the OpenAI backend with Granite Switch,
see [`../granite-switch/`](../granite-switch/).

> **Note:** Not all intrinsics are embedded in every Granite Switch model. The
> current model includes: `answerability`, `citations`, `context_relevance`,
> `guardian-core`, `hallucination_detection`, `query_clarification`,
> `query_rewrite`, and `requirement-check`. Check the model's
> `adapter_index.json` for the full list.

## Available Intrinsics

- **requirement_check**: Validate requirements (used by ALoraRequirement)
- **answerability**: Determine if question is answerable
- **citations**: Extract and validate citations
- **context_relevance**: Assess context-query relevance
- **hallucination_detection**: Detect hallucinated content
- **query_rewrite**: Improve query formulation
- **uncertainty**: Estimate certainty about answering a question
- **policy_guardrails**: Determine if scenario complies with policy
- **guardian-core**: Safety risk detection (harm, bias, groundedness, custom criteria)
- **factuality_detection**: Detect factually incorrect responses
- **factuality_correction**: Correct factually incorrect responses
- **context-attribution**: Identify context sentences that most influenced response

## Related Documentation

- See `mellea/stdlib/components/intrinsic/` for intrinsic implementations
- See `mellea/backends/adapters/` for adapter system
- See `docs/dev/intrinsics_and_adapters.md` for architecture details

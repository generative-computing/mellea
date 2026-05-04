# Intrinsics Examples

This directory contains examples for using Mellea's intrinsic functions - specialized model capabilities accessed through adapters.

<<<<<<< HEAD
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


=======
>>>>>>> main
## Concepts Demonstrated

- **Intrinsic Functions**: Specialized model capabilities beyond text generation
- **Adapter System**: Using LoRA/aLoRA adapters for specific tasks
- **RAG Evaluation**: Assessing retrieval-augmented generation quality
- **Quality Metrics**: Measuring relevance, groundedness, and accuracy
- **Backend Integration**: Adding adapters to different backend types (LocalHFBackend with runtime adapters, OpenAIBackend with Granite Switch embedded adapters)

## Basic Usage

```python
from mellea import model_ids, start_backend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)

response, ctx = mfuncs.chat("What is 2 + 2?", ctx, backend)
print(f"Response: {response.content}")

# NOTE: There are additional functions for other intrinsics as well.
result = core.check_certainty(ctx, backend)
print(f"Certainty score: {result}")
```

OpenAIBackends also support a type of embedded adapter for Granite Switch models:
```python
backend = OpenAIBackend(
        model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
        load_embedded_adapters=True,  # Auto-loads adapters from huggingface repo.
        ...
)
```

The underlying intrinsics can also be utilized directly when generating:
```python
from mellea.stdlib.components import Intrinsic
import mellea.stdlib.functional as mfuncs

...

out, new_ctx = mfuncs.act(
    Intrinsic(
        "requirement-check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."}),
    ctx,
    backend
)
```

For complete runnable examples using the OpenAI backend with Granite Switch,
see [`../granite-switch/`](../granite-switch/).

> **Note:** Not all intrinsics are embedded in every Granite Switch model. You should check
> the model's `adapter_index.json` file for a definitive list. For granite switch models
> pre-built by IBM, we include a list of models in the Mellea `model_id`.

## Available Intrinsics

- **answerability**: Determine if question is answerable
- **citations**: Extract and validate citations
- **context-attribution**: Identify context sentences that most influenced response
- **factuality_correction**: Correct factually incorrect responses
- **factuality_detection**: Detect factually incorrect responses
- **guardian-core**: Safety risk detection (harm, bias, groundedness, custom criteria)
- **hallucination_detection**: Detect hallucinated content
- **policy_guardrails**: Determine if scenario complies with policy
- **query_clarification**: Generate a clarification request if needed, otherwise "CLEAR".
- **query_rewrite**: Improve query formulation
- **requirement_check**: Validate requirements (used by ALoraRequirement)
- **uncertainty**: Estimate certainty about answering a question

## Architecture
![Granite Libraries Software Stack Architecture in Mellea](../../docs/images/granite-libraries-mellea-architecture.png)

## Related Documentation

- See `mellea/stdlib/components/intrinsic/` for intrinsic implementations
- See `mellea/backends/adapters/` for adapter system
- See `docs/dev/intrinsics_and_adapters.md` for architecture details
- See `docs/docs/examples/granite-switch/README.md` for more about granite-switch